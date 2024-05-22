# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
from onnx import TensorProto
from xgboost import XGBClassifier

try:
    from xgboost import XGBRFClassifier
except ImportError:
    XGBRFClassifier = None
from ...common._registration import register_converter
from ..common import get_xgb_params, get_n_estimators_classifier


class XGBConverter:
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
        if base_score is None:
            base_score = 0.5
        booster = xgb_node.get_booster()
        # The json format was available in October 2017.
        # XGBoost 0.7 was the first version released with it.
        js_tree_list = booster.get_dump(with_stats=True, dump_format="json")
        js_trees = [json.loads(s) for s in js_tree_list]
        return objective, base_score, js_trees, best_ntree_limit

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
        attr_pairs["nodes_missing_value_tracks_true"].append(missing)
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
        treeid, tree_weight, jsnode, attr_pairs, is_classifier, remap
    ):
        if "children" in jsnode:
            XGBConverter._add_node(
                attr_pairs=attr_pairs,
                is_classifier=is_classifier,
                tree_id=treeid,
                tree_weight=tree_weight,
                value=jsnode["split_condition"],
                node_id=remap[jsnode["nodeid"]],
                feature_id=jsnode["split"],
                mode="BRANCH_LT",  # 'BRANCH_LEQ' --> is for sklearn
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
                        treeid, tree_weight, ch, attr_pairs, is_classifier, remap
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
                node_id=remap[jsnode["nodeid"]],
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
            XGBConverter._fill_node_attributes(
                treeid, w, jstree, attr_pairs, is_classifier, remap
            )


class XGBRegressorConverter(XGBConverter):
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
        objective, base_score, js_trees, best_ntree_limit = XGBConverter.common_members(
            xgb_node, inputs
        )

        if objective in ["reg:gamma", "reg:tweedie"]:
            raise RuntimeError("Objective '{}' not supported.".format(objective))

        attr_pairs = XGBRegressorConverter._get_default_tree_attribute_pairs()
        attr_pairs["base_values"] = [base_score]

        if best_ntree_limit and best_ntree_limit < len(js_trees):
            js_trees = js_trees[:best_ntree_limit]

        XGBConverter.fill_tree_attributes(
            js_trees, attr_pairs, [1 for _ in js_trees], False
        )

        params = XGBConverter.get_xgb_params(xgb_node)
        attr_pairs["n_targets"] = params["n_targets"]

        # add nodes
        if objective == "count:poisson":
            names = [scope.get_unique_variable_name("tree")]
            del attr_pairs["base_values"]
        else:
            names = operator.output_full_names
        container.add_node(
            "TreeEnsembleRegressor",
            operator.input_full_names,
            names,
            op_domain="ai.onnx.ml",
            name=scope.get_unique_operator_name("TreeEnsembleRegressor"),
            **attr_pairs,
        )

        if objective == "count:poisson":
            cst = scope.get_unique_variable_name("poisson")
            container.add_initializer(cst, TensorProto.FLOAT, [1], [base_score])
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

        objective, base_score, js_trees, best_ntree_limit = XGBConverter.common_members(
            xgb_node, inputs
        )

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

        if ncl <= 1:
            ncl = 2
            if objective != "binary:hinge":
                # See https://github.com/dmlc/xgboost/blob/main/src/common/math.h#L23.
                attr_pairs["post_transform"] = "LOGISTIC"
                attr_pairs["class_ids"] = [0 for v in attr_pairs["class_treeids"]]
                if js_trees[0].get("leaf", None) == 0:
                    attr_pairs["base_values"] = [base_score]
                elif base_score != 0.5:
                    # 0.5 -> cst = 0
                    cst = -np.log(1 / np.float32(base_score) - 1.0)
                    attr_pairs["base_values"] = [cst]
            else:
                attr_pairs["base_values"] = [base_score]
        else:
            # See https://github.com/dmlc/xgboost/blob/main/src/common/math.h#L35.
            attr_pairs["post_transform"] = "SOFTMAX"
            attr_pairs["base_values"] = [base_score for n in range(ncl)]
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
