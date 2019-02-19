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
        
    return params        
