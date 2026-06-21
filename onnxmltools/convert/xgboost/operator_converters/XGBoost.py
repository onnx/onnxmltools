# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from onnx import TensorProto
from xgboost import XGBClassifier
from typing import Any, Dict, List, Union

try:
    from xgboost import XGBRFClassifier
except ImportError:
    XGBRFClassifier = None
from ...common._registration import register_converter
from ..common import get_xgb_params, get_n_estimators_classifier

Node = Dict[str, Any]
TreeLike = Union[Node, List[Node]]


class XGBConverter:
    """
    Base class for converting XGBoost models to ONNX format.
    This class provides methods to validate the model, retrieve parameters,
    and fill in the attributes for the ONNX TreeEnsemble node.
    """

    @staticmethod
    def get_xgb_params(xgb_node):
        """
        Retrieves parameters of a model.
        """
        return get_xgb_params(xgb_node)

    @staticmethod
    def validate(xgb_node):
        params = XGBConverter.get_xgb_params(xgb_node)
        try:
            if "objective" not in params:
                raise AttributeError("ojective")
        except AttributeError as e:
            raise RuntimeError("Missing attribute in XGBoost model " + str(e))
        if hasattr(xgb_node, "missing") and not np.isnan(xgb_node.missing):
            raise RuntimeError(
                "Cannot convert a XGBoost model where missing values are not "
                "nan but {}.".format(xgb_node.missing)
            )

    @staticmethod
    def common_members(xgb_node, inputs):
        params = XGBConverter.get_xgb_params(xgb_node)
        objective = params["objective"]
        base_score = params["base_score"]
        if hasattr(xgb_node, "best_ntree_limit"):
            best_ntree_limit = xgb_node.best_ntree_limit
        elif hasattr(xgb_node, "best_iteration"):
            best_ntree_limit = xgb_node.best_iteration + 1
        else:
            best_ntree_limit = params.get("best_ntree_limit", None)

        # base_score is in probability space regardless of XGBoost version.
        # get_xgb_params() returns it as a list (e.g. [0.5]) whenever it was
        # read from the model config via save_config(), which is the case
        # for essentially every model produced by a real training run; the
        # plain-float fallback below only normalises the rare case where
        # that config key is absent (e.g. a Booster predating JSON config
        # support).
        if isinstance(base_score, list):
            pass
        elif base_score is None:
            base_score = [0.5]
        else:
            base_score = [float(base_score)]

        booster = xgb_node.get_booster()
        # The json format was available in October 2017.
        # XGBoost 0.7 was the first version released with it.
        js_tree_list = booster.get_dump(with_stats=True, dump_format="json")
        js_trees: TreeLike = [json.loads(s) for s in js_tree_list]
        js_trees = XGBConverter._process_categorical_features(js_trees)
        return objective, base_score, js_trees, best_ntree_limit

    @staticmethod
    def _is_bracketed_json_list_string(s: str) -> bool:
        s = s.strip()
        return len(s) >= 2 and s[0] == "[" and s[-1] == "]"

    @staticmethod
    def _maybe_transform_categorical(node: Node, last_node_id) -> tuple[int, bool]:
        """
        If node's split_condition is a JSON list string, transform it into a
        chain of BRANCH_EQ nodes in-place.
        """

        split_condition = node.get("split_condition")

        if not isinstance(split_condition, list):
            return (last_node_id, False)  # not categorical

        if len(split_condition) == 0:
            raise ValueError("split_condition is an empty array. ")

        # Validate it's a split node with two children
        children = node.get("children")
        if not (isinstance(children, list) and len(children) == 2):
            raise ValueError(
                "Expected a split node with two children before categorical transform."
            )

        orig_left, orig_right = children

        # First category goes on the original node
        node["decision_type"] = "BRANCH_EQ"
        node["split_condition"] = split_condition[0]

        yes_left = orig_left["nodeid"] == node["yes"]

        current_node = node
        for cat in split_condition[1:]:
            new_node = current_node.copy()
            new_node["split_condition"] = cat
            last_node_id += 1
            new_node["nodeid"] = last_node_id

            if current_node["missing"] == current_node["no"]:
                current_node["missing"] = new_node["nodeid"]
            current_node["no"] = new_node["nodeid"]
            if yes_left:
                current_node["children"] = [orig_left, new_node]
            else:
                current_node["children"] = [new_node, orig_right]
            current_node = new_node

        # Final "no" path goes to the original right subtree
        current_node["children"] = [orig_left, orig_right]
        return (last_node_id, True)

    @staticmethod
    def _process_node(node: Node, last_node_id: int) -> int:
        # If this is a leaf, nothing to do
        if "children" not in node or not isinstance(node["children"], list):
            return last_node_id

        for child in node["children"]:
            last_node_id = XGBConverter._process_node(child, last_node_id)

        last_node_id, transformed = XGBConverter._maybe_transform_categorical(
            node, last_node_id
        )
        if not transformed:
            # Non-categorical split node: enforce BRANCH_LT as default
            node["decision_type"] = "BRANCH_LT"
        return last_node_id

    @staticmethod
    def _process_root(root: Node) -> None:
        last_node_id = XGBConverter._find_last_node_id(root)
        XGBConverter._process_node(root, last_node_id)

    @staticmethod
    def _find_last_node_id(node: Node) -> int:
        if "children" not in node:
            return node["nodeid"]

        max_id = node["nodeid"]
        for child in node["children"]:
            child_max = XGBConverter._find_last_node_id(child)
            if child_max > max_id:
                max_id = child_max

        return max_id

    @staticmethod
    def _process_categorical_features(js_tree: TreeLike) -> TreeLike:
        """
        Processes the native handling of categorical features to equality checks that
        are supported in Onnx.

        - If a split node encodes categories via a JSON list string in 'split_condition',
        it is expanded into a chain of BRANCH_EQ nodes.
        - Otherwise (non-categorical split), the node's 'decision_type' is set to 'BRANCH_LT'.
        - If there are categorical features, the nodeids are updated, but depth is ignored
        since its not used for the conversion

        Returns the processed tree model.
        """
        if isinstance(js_tree, list):
            for root in js_tree:
                if isinstance(root, dict):
                    XGBConverter._process_root(root)
        elif isinstance(js_tree, dict):
            XGBConverter._process_root(js_tree)
        else:
            raise TypeError(
                "js_tree must be a dict (single tree) or list of dicts (forest)."
            )
        return js_tree

    @staticmethod
    def _get_default_tree_attribute_pairs(is_classifier):
        attrs = {}
        for k in {
            "nodes_treeids",
            "nodes_nodeids",
            "nodes_featureids",
            "nodes_modes",
            "nodes_values",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
        }:
            attrs[k] = []
        if is_classifier:
            for k in {"class_treeids", "class_nodeids", "class_ids", "class_weights"}:
                attrs[k] = []
        else:
            for k in {
                "target_treeids",
                "target_nodeids",
                "target_ids",
                "target_weights",
            }:
                attrs[k] = []
        return attrs

    @staticmethod
    def _add_node(
        attr_pairs,
        is_classifier,
        tree_id,
        tree_weight,
        node_id,
        feature_id,
        mode,
        value,
        true_child_id,
        false_child_id,
        weights,
        weight_id_bias,
        missing,
        hitrate,
    ):
        if isinstance(feature_id, str):
            # Something like f0, f1...
            if feature_id[0] == "f":
                try:
                    feature_id = int(feature_id[1:])
                except ValueError:
                    raise RuntimeError(
                        "Unable to interpret '{0}', feature "
                        "names should follow pattern 'f%d'.".format(feature_id)
                    )
            else:
                try:
                    feature_id = int(float(feature_id))
                except ValueError:
                    raise RuntimeError(
                        "Unable to interpret '{0}', feature "
                        "names should follow pattern 'f%d'.".format(feature_id)
                    )

        # Split condition for sklearn
        # * if X_ptr[X_sample_stride * i + X_fx_stride * node.feature] <= node.threshold:
        # * https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx#L946
        # Split condition for xgboost
        # * if (fvalue < split_value)
        # * https://github.com/dmlc/xgboost/blob/main/include/xgboost/tree_model.h#L804

        attr_pairs["nodes_treeids"].append(tree_id)
        attr_pairs["nodes_nodeids"].append(node_id)
        attr_pairs["nodes_featureids"].append(feature_id)
        attr_pairs["nodes_modes"].append(mode)
        attr_pairs["nodes_values"].append(float(value))
        attr_pairs["nodes_truenodeids"].append(true_child_id)
        attr_pairs["nodes_falsenodeids"].append(false_child_id)
        attr_pairs["nodes_missing_value_tracks_true"].append(int(missing))
        if "nodes_hitrates" in attr_pairs:
            attr_pairs["nodes_hitrates"].append(hitrate)
        if mode == "LEAF":
            if is_classifier:
                for i, w in enumerate(weights):
                    attr_pairs["class_treeids"].append(tree_id)
                    attr_pairs["class_nodeids"].append(node_id)
                    attr_pairs["class_ids"].append(i + weight_id_bias)
                    attr_pairs["class_weights"].append(float(tree_weight * w))
            else:
                for i, w in enumerate(weights):
                    attr_pairs["target_treeids"].append(tree_id)
                    attr_pairs["target_nodeids"].append(node_id)
                    attr_pairs["target_ids"].append(i + weight_id_bias)
                    attr_pairs["target_weights"].append(float(tree_weight * w))

    @staticmethod
    def _fill_node_attributes(
        treeid, tree_weight, jsnode, attr_pairs, is_classifier, remap, ids_covered: set
    ):
        node_id = remap[jsnode["nodeid"]]
        if node_id in ids_covered:
            return
        else:
            ids_covered.add(node_id)

        if "children" in jsnode:
            XGBConverter._add_node(
                attr_pairs=attr_pairs,
                is_classifier=is_classifier,
                tree_id=treeid,
                tree_weight=tree_weight,
                value=jsnode["split_condition"],
                node_id=node_id,
                feature_id=jsnode["split"],
                mode=jsnode["decision_type"],  # 'BRANCH_LEQ' --> is for sklearn
                # 'BRANCH_LT' --> is for xgboost numerical features
                # 'BRANCH_EQ' --> is for xgboost categorical features
                true_child_id=remap[jsnode["yes"]],  # ['children'][0]['nodeid'],
                false_child_id=remap[jsnode["no"]],  # ['children'][1]['nodeid'],
                weights=None,
                weight_id_bias=None,
                missing=jsnode.get("missing", -1) == jsnode["yes"],
                hitrate=jsnode.get("cover", 0),
            )

            for ch in jsnode["children"]:
                if "children" in ch or "leaf" in ch:
                    XGBConverter._fill_node_attributes(
                        treeid,
                        tree_weight,
                        ch,
                        attr_pairs,
                        is_classifier,
                        remap,
                        ids_covered,
                    )
                else:
                    raise RuntimeError("Unable to convert this node {0}".format(ch))

        else:
            weights = [jsnode["leaf"]]
            weights_id_bias = 0
            XGBConverter._add_node(
                attr_pairs=attr_pairs,
                is_classifier=is_classifier,
                tree_id=treeid,
                tree_weight=tree_weight,
                value=0.0,
                node_id=node_id,
                feature_id=0,
                mode="LEAF",
                true_child_id=0,
                false_child_id=0,
                weights=weights,
                weight_id_bias=weights_id_bias,
                missing=False,
                hitrate=jsnode.get("cover", 0),
            )

    @staticmethod
    def _remap_nodeid(jsnode, remap=None):
        if remap is None:
            remap = {}
        nid = jsnode["nodeid"]
        if nid not in remap:
            remap[nid] = len(remap)
        if "children" in jsnode:
            for ch in jsnode["children"]:
                XGBConverter._remap_nodeid(ch, remap)
        return remap

    @staticmethod
    def fill_tree_attributes(js_xgb_node, attr_pairs, tree_weights, is_classifier):
        if not isinstance(js_xgb_node, list):
            raise TypeError("js_xgb_node must be a list")
        for treeid, (jstree, w) in enumerate(zip(js_xgb_node, tree_weights)):
            remap = XGBConverter._remap_nodeid(jstree)
            ids_covered = set()
            XGBConverter._fill_node_attributes(
                treeid, w, jstree, attr_pairs, is_classifier, remap, ids_covered
            )


def _compute_base_score_logit(base_score):
    """
    Convert a base_score probability value to logit space.
    Returns (logit_value, is_zero) where is_zero=True means logit is 0
    (i.e. base_score == 0.5) and no base_values entry is needed.
    """
    bs_val = np.float32(base_score)
    bs_clipped = np.clip(bs_val, 1e-7, 1.0 - 1e-7)
    logit_bs = float(-np.log(1.0 / bs_clipped - 1.0))
    return logit_bs, np.isclose(logit_bs, 0.0)


class XGBRegressorConverter(XGBConverter):
    """
    Converter for XGBoost Regressor models to ONNX format.
    This class inherits from XGBConverter and implements the conversion
    logic specific to regression tasks.
    It handles the conversion of model parameters, tree structure,
    and the creation of the ONNX TreeEnsembleRegressor node.
    """

    @staticmethod
    def validate(xgb_node):
        return XGBConverter.validate(xgb_node)

    @staticmethod
    def _get_default_tree_attribute_pairs():
        attrs = XGBConverter._get_default_tree_attribute_pairs(False)
        attrs["post_transform"] = "NONE"
        attrs["n_targets"] = 1
        return attrs

    @staticmethod
    def convert(scope, operator, container):
        xgb_node = operator.raw_operator
        inputs = operator.inputs
        (
            objective,
            base_score,
            js_trees,
            best_ntree_limit,
        ) = XGBConverter.common_members(xgb_node, inputs)

        # base_score is always a list at this point (normalised in common_members)
        bs_list = base_score

        if best_ntree_limit and best_ntree_limit < len(js_trees):
            js_trees = js_trees[:best_ntree_limit]

        attr_pairs = XGBRegressorConverter._get_default_tree_attribute_pairs()
        XGBConverter.fill_tree_attributes(
            js_trees, attr_pairs, [1 for _ in js_trees], False
        )

        params = XGBConverter.get_xgb_params(xgb_node)
        attr_pairs["n_targets"] = params["n_targets"]

        # binary:logistic: XGBoost accumulates tree outputs in logit space and
        # base_score is stored in probability space (in both XGBoost <2 and
        # >=2), so it must be converted to logit space before being added to
        # the tree sum.
        if objective == "binary:logistic":
            bs_val = np.float32(bs_list[0])
            if not (0.0 < bs_val < 1.0):
                raise ValueError(
                    f"base_score={bs_val} is out of range for binary:logistic; "
                    "expected a probability in (0, 1)."
                )
            logit_bs, is_zero = _compute_base_score_logit(bs_val)
            if is_zero:
                attr_pairs.pop("base_values", None)
            else:
                attr_pairs["base_values"] = [logit_bs]

            raw_name = scope.get_unique_variable_name("binary_logistic_raw")
            container.add_node(
                "TreeEnsembleRegressor",
                operator.input_full_names,
                [raw_name],
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
                **attr_pairs,
            )
            container.add_node(
                "Sigmoid",
                [raw_name],
                operator.output_full_names,
                name=scope.get_unique_operator_name("Sigmoid"),
            )
            return

        # add nodes
        objectives_with_loglink = {"count:poisson", "reg:gamma", "reg:tweedie"}
        if objective in objectives_with_loglink:
            names = [scope.get_unique_variable_name("tree")]
            attr_pairs.pop("base_values", None)
        else:
            attr_pairs["base_values"] = bs_list
            names = operator.output_full_names

        container.add_node(
            "TreeEnsembleRegressor",
            operator.input_full_names,
            names,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
            **attr_pairs,
        )

        if objective in objectives_with_loglink:
            cst = scope.get_unique_variable_name("raw_prediction")
            container.add_initializer(
                cst, TensorProto.FLOAT, [len(bs_list)], bs_list
            )
            new_name = scope.get_unique_variable_name("exp")
            container.add_node("Exp", names, [new_name])
            container.add_node("Mul", [new_name, cst], operator.output_full_names)


class XGBClassifierConverter(XGBConverter):
    @staticmethod
    def validate(xgb_node):
        return XGBConverter.validate(xgb_node)

    @staticmethod
    def _get_default_tree_attribute_pairs():
        attrs = XGBConverter._get_default_tree_attribute_pairs(True)
        # TODO: check it is implemented. The model cannot
        # be loaded when they are present.
        # attrs['nodes_hitrates'] = []
        return attrs

    @staticmethod
    def convert(scope, operator, container):
        xgb_node = operator.raw_operator
        inputs = operator.inputs

        (
            objective,
            base_score,
            js_trees,
            best_ntree_limit,
        ) = XGBConverter.common_members(xgb_node, inputs)

        # base_score is always a list at this point (normalised in common_members)

        params = XGBConverter.get_xgb_params(xgb_node)
        n_estimators = get_n_estimators_classifier(xgb_node, params, js_trees)
        num_class = params.get("num_class", None)

        attr_pairs = XGBClassifierConverter._get_default_tree_attribute_pairs()
        XGBConverter.fill_tree_attributes(
            js_trees, attr_pairs, [1 for _ in js_trees], True
        )
        if num_class is not None:
            ncl = num_class
            n_estimators = len(js_trees) // ncl
        else:
            ncl = (max(attr_pairs["class_treeids"]) + 1) // n_estimators

        best_ntree_limit = best_ntree_limit or len(js_trees)
        if ncl > 0:
            best_ntree_limit *= ncl
        if 0 < best_ntree_limit < len(js_trees):
            js_trees = js_trees[:best_ntree_limit]
            attr_pairs = XGBClassifierConverter._get_default_tree_attribute_pairs()
            XGBConverter.fill_tree_attributes(
                js_trees, attr_pairs, [1 for _ in js_trees], True
            )

        if len(attr_pairs["class_treeids"]) == 0:
            raise RuntimeError("XGBoost model is empty.")

        all_zero_weights = False
        if ncl <= 1:
            ncl = 2
            if objective != "binary:hinge":
                # See https://github.com/dmlc/xgboost/blob/main/src/common/math.h#L23.
                attr_pairs["class_ids"] = [0 for v in attr_pairs["class_treeids"]]
                all_zero_weights = all(
                    w == 0.0 for w in attr_pairs["class_weights"]
                )
                if all_zero_weights:
                    # Degenerate model: every leaf is exactly zero, so the
                    # prediction is a constant fully determined by
                    # base_score. onnxruntime's handling of
                    # TreeEnsembleClassifier with post_transform=LOGISTIC and
                    # all-zero class_weights has been observed to differ by
                    # platform/CPU for the same onnxruntime version, so we
                    # synthesize explicit per-class weights with
                    # post_transform=NONE instead, which is stable. Its
                    # native label output still breaks an exact 0.5/0.5 tie
                    # towards the higher class index (the opposite of
                    # XGBoost's tiebreak), so the predicted label is
                    # recomputed below via ArgMax+Gather.
                    bs_val = float(np.clip(base_score[0], 1e-7, 1.0 - 1e-7))
                    p1 = bs_val
                    p0 = 1.0 - p1
                    attr_pairs["post_transform"] = "NONE"
                    attr_pairs.pop("base_values", None)
                    first_node = attr_pairs["class_nodeids"][0]
                    attr_pairs["class_treeids"] = [0, 0]
                    attr_pairs["class_nodeids"] = [first_node, first_node]
                    attr_pairs["class_ids"] = [0, 1]
                    attr_pairs["class_weights"] = [p0, p1]
                else:
                    # XGBoost accumulates tree outputs in logit space and
                    # base_score is stored in probability space (in both
                    # XGBoost <2 and >=2), so it must be converted to logit
                    # space before being added to the tree sum.
                    attr_pairs["post_transform"] = "LOGISTIC"
                    bs_val = float(base_score[0])
                    logit_bs, is_zero = _compute_base_score_logit(bs_val)
                    if is_zero:
                        attr_pairs.pop("base_values", None)
                    else:
                        attr_pairs["base_values"] = [logit_bs]
            else:
                attr_pairs["base_values"] = base_score
        else:
            # See https://github.com/dmlc/xgboost/blob/main/src/common/math.h#L35.
            attr_pairs["post_transform"] = "SOFTMAX"
            # If base_score has fewer elements than classes, replicate to match
            if len(base_score) == 1:
                attr_pairs["base_values"] = base_score * ncl
            else:
                attr_pairs["base_values"] = base_score
            attr_pairs["class_ids"] = [v % ncl for v in attr_pairs["class_treeids"]]

        classes = xgb_node.classes_
        if (
            np.issubdtype(classes.dtype, np.floating)
            or np.issubdtype(classes.dtype, np.integer)
            or np.issubdtype(classes.dtype, np.bool_)
        ):
            attr_pairs["classlabels_int64s"] = classes.astype("int")
        else:
            classes = np.array([s.encode("utf-8") for s in classes])
            attr_pairs["classlabels_strings"] = classes

        # add nodes
        if objective in ("binary:logistic", "binary:hinge"):
            ncl = 2
            if objective == "binary:hinge":
                attr_pairs["post_transform"] = "NONE"
                output_names = [
                    operator.output_full_names[0],
                    scope.get_unique_variable_name("output_prob"),
                ]
            elif all_zero_weights:
                output_names = [
                    scope.get_unique_variable_name("xgb_raw_label"),
                    operator.output_full_names[1],
                ]
            else:
                output_names = operator.output_full_names
            container.add_node(
                "TreeEnsembleClassifier",
                operator.input_full_names,
                output_names,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("TreeEnsembleClassifier"),
                **attr_pairs,
            )
            if objective == "binary:hinge":
                if container.target_opset < 9:
                    raise RuntimeError(
                        f"hinge function cannot be implemented because "
                        f"opset={container.target_opset}<9."
                    )
                zero = scope.get_unique_variable_name("zero")
                one = scope.get_unique_variable_name("one")
                container.add_initializer(zero, TensorProto.FLOAT, [1], [0.0])
                container.add_initializer(one, TensorProto.FLOAT, [1], [1.0])
                greater = scope.get_unique_variable_name("output_prob")
                container.add_node("Greater", [output_names[1], zero], [greater])
                container.add_node(
                    "Where", [greater, one, zero], operator.output_full_names[1]
                )
            elif all_zero_weights:
                # ArgMax's default tiebreak (first/lowest index on ties)
                # matches XGBoost's, unlike TreeEnsembleClassifier's own
                # label output in this degenerate case.
                argmax_name = scope.get_unique_variable_name("xgb_argmax")
                container.add_node(
                    "ArgMax",
                    [operator.output_full_names[1]],
                    [argmax_name],
                    axis=1,
                    keepdims=0,
                    name=scope.get_unique_operator_name("ArgMax"),
                )
                labels_name = scope.get_unique_variable_name("xgb_classlabels")
                if "classlabels_int64s" in attr_pairs:
                    container.add_initializer(
                        labels_name,
                        TensorProto.INT64,
                        [len(attr_pairs["classlabels_int64s"])],
                        [int(c) for c in attr_pairs["classlabels_int64s"]],
                    )
                else:
                    container.add_initializer(
                        labels_name,
                        TensorProto.STRING,
                        [len(attr_pairs["classlabels_strings"])],
                        list(attr_pairs["classlabels_strings"]),
                    )
                container.add_node(
                    "Gather",
                    [labels_name, argmax_name],
                    [operator.output_full_names[0]],
                    name=scope.get_unique_operator_name("Gather"),
                )
        elif objective in ("multi:softprob", "multi:softmax"):
            ncl = len(js_trees) // n_estimators
            if objective == "multi:softmax":
                attr_pairs["post_transform"] = "NONE"
            container.add_node(
                "TreeEnsembleClassifier",
                operator.input_full_names,
                operator.output_full_names,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("TreeEnsembleClassifier"),
                **attr_pairs,
            )
        elif objective == "reg:logistic":
            ncl = len(js_trees) // n_estimators
            if ncl == 1:
                ncl = 2
            container.add_node(
                "TreeEnsembleClassifier",
                operator.input_full_names,
                operator.output_full_names,
                op_domain="ai.onnx.ml",
                name=scope.get_unique_operator_name("TreeEnsembleClassifier"),
                **attr_pairs,
            )
        else:
            raise RuntimeError("Unexpected objective: {0}".format(objective))


def convert_xgboost(scope, operator, container):
    """
    Converts an XGBoost model (XGBClassifier or XGBRegressor) into an ONNX TreeEnsemble node.

    Parameters:
        scope: Object for managing variable names in the ONNX graph.
        operator: Wrapper for the XGBoost model and its input/output variables.
        container: Object to which the ONNX nodes will be added.

    This function dispatches the conversion to the appropriate internal converter
    based on whether the model is a classifier or regressor.
    """
    xgb_node = operator.raw_operator
    if isinstance(xgb_node, (XGBClassifier, XGBRFClassifier)) or getattr(
        xgb_node, "operator_name", None
    ) in ("XGBClassifier", "XGBRFClassifier"):
        cls = XGBClassifierConverter
    else:
        cls = XGBRegressorConverter
    cls.validate(xgb_node)
    cls.convert(scope, operator, container)


register_converter("XGBClassifier", convert_xgboost)
register_converter("XGBRFClassifier", convert_xgboost)
register_converter("XGBRegressor", convert_xgboost)
register_converter("XGBRFRegressor", convert_xgboost)