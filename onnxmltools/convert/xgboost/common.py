# SPDX-License-Identifier: Apache-2.0

"""
Common function to converters and shape calculators.
"""

def get_xgb_params(xgb_node):
    """
    Retrieves parameters of a model.
    """
    if hasattr(xgb_node, 'kwargs'):
        # XGBoost >= 0.7
        params = xgb_node.get_xgb_params()
    else:
        # XGBoost < 0.7
        params = xgb_node.__dict__

    if ('n_estimators' not in params and
            hasattr(xgb_node, 'n_estimators')):
        # xgboost >= 1.0.2
        params['n_estimators'] = xgb_node.n_estimators
    return params
