# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType, Int64TensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sklearn_scaler_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C_1], ..., [N, C_n] ---> [N, C_1 + ... + C_n]

    Similar to imputer, this operator can take multiple input feature tensors and concatenate them along C-axis.
    '''
    check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType],
                                 good_output_types=[FloatTensorType])
    # Inputs: multiple float- and integer-tensors
    # Output: one float tensor
    for variable in operator.inputs:
        if len(variable.type.shape) != 2:
            raise RuntimeError('Only 2-D tensor(s) can be input(s)')
        if len(set(variable.type.shape[0] for variable in operator.inputs)) > 1:
            raise RuntimeError('Batch size must be identical across inputs')

    N = operator.inputs[0].type.shape[0]
    C = 0
    for variable in operator.inputs:
        if isinstance(variable.type.shape[1], numbers.Integral):
            C += variable.type.shape[1]
        else:
            C = 'None'
            break

    operator.outputs[0].type.shape = [N, C]


register_shape_calculator('SklearnRobustScaler', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnScaler', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnNormalizer', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnMinMaxScaler', calculate_sklearn_scaler_output_shapes)
register_shape_calculator('SklearnMaxAbsScaler', calculate_sklearn_scaler_output_shapes)
