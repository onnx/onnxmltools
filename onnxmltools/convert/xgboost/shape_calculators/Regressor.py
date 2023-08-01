# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_regressor_output_shapes

register_shape_calculator('XGBRegressor', calculate_linear_regressor_output_shapes)
