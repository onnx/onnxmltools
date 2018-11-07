# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import calculate_linear_regressor_output_shapes


register_shape_calculator('SklearnElasticNetRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnLinearRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnLinearSVR', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnDecisionTreeRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnRandomForestRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreesRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnGradientBoostingRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnKNeighborsRegressor', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnLassoLars', calculate_linear_regressor_output_shapes)
register_shape_calculator('SklearnRidge', calculate_linear_regressor_output_shapes)
