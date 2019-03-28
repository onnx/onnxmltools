# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_merge_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C_1, H_1, W_1], ..., [N, C_n, H_n, W_n] --->
            [N, max(C_1, ..., C_n), max(H_1, ..., H_n), max(W_1, ..., W_n)]

    If 'None' happens at any coordinate, that coordinate's final dimension would be 'None'.
    '''
    check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    # [TODO] Fix reduce-like shape inference. We now assume all inputs are 4-D.
    n_dims = max(len(variable.type.shape) for variable in operator.inputs)
    output_shape = [0] * n_dims
    for i in range(n_dims):
        input_dims = [variable.type.shape[i] for variable in operator.inputs if len(variable.type.shape) > i]
        if 'None' in input_dims:
            output_shape[i] = 'None'
        else:
            output_shape[i] = max(input_dims)

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('add', calculate_merge_output_shapes)
register_shape_calculator('average', calculate_merge_output_shapes)
register_shape_calculator('max', calculate_merge_output_shapes)
register_shape_calculator('min', calculate_merge_output_shapes)
register_shape_calculator('multiply', calculate_merge_output_shapes)
