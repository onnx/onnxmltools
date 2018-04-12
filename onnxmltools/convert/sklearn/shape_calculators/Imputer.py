# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_imputer_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])

    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] != 'None':
            C += variable.type.shape[1]
        else:
            C = 'None'
            break

    operator.outputs[0].type = FloatTensorType([1, C])


register_shape_calculator('SklearnImputer', calculate_sklearn_imputer_output_shapes)
register_shape_calculator('SklearnBinarizer', calculate_sklearn_imputer_output_shapes)
