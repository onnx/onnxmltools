# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType
from ..common import get_xgb_params


def calculate_xgboost_regressor_output_shapes(operator):
    N = operator.inputs[0].type.shape[0]
    n_targets = get_xgb_params(operator.raw_operator).get("n_targets", 1)
    operator.outputs[0].type = FloatTensorType([N, n_targets])


register_shape_calculator("XGBRegressor", calculate_xgboost_regressor_output_shapes)
register_shape_calculator("XGBRFRegressor", calculate_xgboost_regressor_output_shapes)
