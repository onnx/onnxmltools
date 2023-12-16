# SPDX-License-Identifier: Apache-2.0

import json
import re
import pprint
from packaging.version import Version
import numpy as np
from xgboost import XGBRegressor, XGBClassifier, __version__

try:
    from xgboost import XGBRFRegressor, XGBRFClassifier
except ImportError:
    # old version of xgboost
    XGBRFRegressor, XGBRFClassifier = None, None
from onnxconverter_common.data_types import FloatTensorType
from ..common._container import XGBoostModelContainer
from ..common._topology import Topology


xgboost_classifier_list = [XGBClassifier]

# Associate types with our operator names.
xgboost_operator_name_map = {
    XGBClassifier: "XGBClassifier",
    XGBRegressor: "XGBRegressor",
}

if XGBRFClassifier:
    xgboost_operator_name_map.update(
        {
            XGBRFClassifier: "XGBRFClassifier",
            XGBRFRegressor: "XGBRFRegressor",
        }
    )
    xgboost_classifier_list.append(XGBRFClassifier)


def _append_covers(node):
    res = []
    if "cover" in node:
        res.append(node["cover"])
    if "children" in node:
        for ch in node["children"]:
            res.extend(_append_covers(ch))
    return res


def _get_attributes(booster):
    atts = booster.attributes()
    dp = booster.get_dump(dump_format="json", with_stats=True)
    res = [json.loads(d) for d in dp]

    # num_class
    if Version(__version__) < Version("1.5"):
        state = booster.__getstate__()
        bstate = bytes(state["handle"])
        reg = re.compile(b'("tree_info":\\[[0-9,]*\\])')
        objs = list(set(reg.findall(bstate)))
        if len(objs) != 1:
            raise RuntimeError(
                "Unable to retrieve the tree coefficients from\n%s"
                "" % bstate.decode("ascii", errors="ignore")
            )
        tree_info = json.loads("{{{}}}".format(objs[0].decode("ascii")))["tree_info"]
        num_class = len(set(tree_info))
        trees = len(res)
        try:
            ntrees = booster.best_ntree_limit
        except AttributeError:
            ntrees = trees // num_class if num_class > 0 else trees
    else:
        trees = len(res)
        ntrees = getattr(booster, "best_ntree_limit", trees)
        config = json.loads(booster.save_config())["learner"]["learner_model_param"]
        num_class = int(config["num_class"]) if "num_class" in config else 0
        if num_class == 0 and ntrees > 0:
            num_class = trees // ntrees
        if num_class == 0:
            raise RuntimeError(
                f"Unable to retrieve the number of classes, num_class={num_class}, "
                f"trees={trees}, ntrees={ntrees}, config={config}."
            )

    kwargs = atts.copy()
    kwargs["feature_names"] = booster.feature_names
    kwargs["n_estimators"] = ntrees

    # covers
    covs = []
    for tr in res:
        covs.extend(_append_covers(tr))

    if all(map(lambda x: int(x) == x, set(covs))):
        # regression
        kwargs["num_target"] = num_class
        kwargs["num_class"] = 0
        kwargs["objective"] = "reg:squarederror"
    else:
        # classification
        kwargs["num_class"] = num_class
        if num_class != 1:
            if Version(__version__) < Version("1.5"):
                reg = re.compile(b"(multi:[a-z]{1,15})")
                objs = list(set(reg.findall(bstate)))
                if len(objs) == 1:
                    kwargs["objective"] = objs[0].decode("ascii")
                else:
                    raise RuntimeError(
                        "Unable to guess objective in %r (trees=%r, ntrees=%r, num_class=%r)"
                        "." % (objs, trees, ntrees, kwargs["num_class"])
                    )
            else:
                att = json.loads(booster.save_config())
                kwargs["objective"] = att["learner"]["objective"]["name"]
                nc = int(att["learner"]["learner_model_param"]["num_class"])
                if nc != num_class:
                    raise RuntimeError(
                        "Mismatched value %r != %r from\n%s"
                        % (nc, num_class, pprint.pformat(att))
                    )
        else:
            kwargs["objective"] = "binary:logistic"

    if "base_score" not in kwargs:
        kwargs["base_score"] = 0.5
    elif isinstance(kwargs["base_score"], str):
        kwargs["base_score"] = float(kwargs["base_score"])
    return kwargs


class WrappedBooster:
    def __init__(self, booster):
        self.booster_ = booster
        self.kwargs = _get_attributes(booster)

        if self.kwargs["num_class"] > 0:
            self.classes_ = self._generate_classes(self.kwargs)
            self.operator_name = "XGBClassifier"
        else:
            self.operator_name = "XGBRegressor"

    def get_xgb_params(self):
        return {k: v for k, v in self.kwargs.items() if v is not None}

    def get_booster(self):
        return self.booster_

    def _generate_classes(self, model_dict):
        if model_dict["num_class"] == 1:
            return np.asarray([0, 1])
        return np.arange(model_dict["num_class"])


def _get_xgboost_operator_name(model):
    """
    Get operator name of the input argument

    :param model_type:  A xgboost object.
    :return: A string which stands for the type of the input model in our conversion framework
    """
    if isinstance(model, WrappedBooster):
        return model.operator_name
    if type(model) not in xgboost_operator_name_map:
        raise ValueError("No proper operator name found for '%s'" % type(model))
    return xgboost_operator_name_map[type(model)]


def _parse_xgboost_simple_model(scope, model, inputs):
    """
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A xgboost object
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    """
    this_operator = scope.declare_local_operator(
        _get_xgboost_operator_name(model), model
    )
    this_operator.inputs = inputs

    if type(model) in xgboost_classifier_list or getattr(
        model, "operator_name", None
    ) in ("XGBClassifier", "XGBRFClassifier"):
        # For classifiers, we may have two outputs, one for label and
        # the other one for probabilities of all classes.
        # Notice that their types here are not necessarily correct
        # and they will be fixed in shape inference phase
        label_variable = scope.declare_local_variable("label", FloatTensorType())
        probability_map_variable = scope.declare_local_variable(
            "probabilities", FloatTensorType()
        )
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_map_variable)
    else:
        # We assume that all scikit-learn operator can only produce a single float tensor.
        variable = scope.declare_local_variable("variable", FloatTensorType())
        this_operator.outputs.append(variable)
    return this_operator.outputs


def _parse_xgboost(scope, model, inputs):
    """
    This is a delegate function. It doesn't nothing but invoke
    the correct parsing function according to the input
    model's type.
    :param scope: Scope object
    :param model: A xgboost object
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    """
    return _parse_xgboost_simple_model(scope, model, inputs)


def parse_xgboost(
    model,
    initial_types=None,
    target_opset=None,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
):
    raw_model_container = XGBoostModelContainer(model)
    topology = Topology(
        raw_model_container,
        default_batch_size="None",
        initial_types=initial_types,
        target_opset=target_opset,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
    )
    scope = topology.declare_scope("__root__")

    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    for variable in inputs:
        raw_model_container.add_input(variable)

    outputs = _parse_xgboost(scope, model, inputs)

    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology
