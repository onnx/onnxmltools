# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers


def calculate_sklearn_linear_regressor_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a batch. If the input batch size is N, the output
    shape may be [N, 1].
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType([N, 1])


register_shape_calculator('SklearnElasticNetRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnLinearRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnLinearSVR', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnDecisionTreeRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnRandomForestRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnExtraTreesRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('SklearnGradientBoostingRegressor', calculate_sklearn_linear_regressor_output_shapes)
register_shape_calculator('LgbmRegressor', calculate_sklearn_linear_regressor_output_shapes)
