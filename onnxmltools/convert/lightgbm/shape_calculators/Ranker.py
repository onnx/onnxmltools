# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_shape_calculator
from .Regressor import calculate_lightgbm_regressor_output_shapes

register_shape_calculator("LgbmRanker", calculate_lightgbm_regressor_output_shapes)

