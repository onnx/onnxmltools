# SPDX-License-Identifier: Apache-2.0

"""
Common function to converters and shape calculators.
"""
import json


def get_xgb_params(xgb_node):
    """
    Retrieves parameters of a model.
    """
    if hasattr(xgb_node, "kwargs"):
        # XGBoost >= 0.7
        params = xgb_node.get_xgb_params()
    else:
        # XGBoost < 0.7
        params = xgb_node.__dict__
    if hasattr("xgb_node", "save_config"):
        config = json.loads(xgb_node.save_config())
    else:
        config = json.loads(xgb_node.get_booster().save_config())
    params = {k: v for k, v in params.items() if v is not None}
    num_class = int(config["learner"]["learner_model_param"]["num_class"])
    if num_class > 0:
        params["num_class"] = num_class
    if "n_estimators" not in params and hasattr(xgb_node, "n_estimators"):
        # xgboost >= 1.0.2
        if xgb_node.n_estimators is not None:
            params["n_estimators"] = xgb_node.n_estimators
    if "base_score" in config["learner"]["learner_model_param"]:
        bs = float(config["learner"]["learner_model_param"]["base_score"])
        # xgboost >= 2.0
        params["base_score"] = bs
    if "num_target" in config["learner"]["learner_model_param"]:
        params["n_targets"] = int(
            config["learner"]["learner_model_param"]["num_target"]
        )
    else:
        params["n_targets"] = 1

    bst = xgb_node.get_booster()
    if hasattr(bst, "best_ntree_limit"):
        params["best_ntree_limit"] = bst.best_ntree_limit
    if "gradient_booster" in config["learner"]:
        gbp = config["learner"]["gradient_booster"]["gbtree_model_param"]
        if "num_trees" in gbp:
            params["best_ntree_limit"] = int(gbp["num_trees"])
    return params


def get_n_estimators_classifier(xgb_node, params, js_trees):
    if "n_estimators" not in params:
        num_class = params.get("num_class", 0)
        if num_class == 0:
            return len(js_trees)
        return len(js_trees) // num_class
    return params["n_estimators"]
