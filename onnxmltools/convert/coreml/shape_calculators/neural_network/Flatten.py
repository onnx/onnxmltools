# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_flatten_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, C]
        2. [N, C, H, W] ---> [N, C * H * W]
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input = operator.inputs[0]
    output = operator.outputs[0]

    if len(input.type.shape) not in [2, 4]:
        raise RuntimeError('Input must be 2-D or 4-D float tensor')

    input_shape = input.type.shape
    output_shape = [input_shape[0], 1]

    # Calculate the multiplication of C, H, and W.
    for i in input_shape[1:]:
        if i != 'None':
            output_shape[1] *= i
        else:
            # If any of C, H, W-dimensions is unknown, the flatten C-dimension is unknown
            output_shape[1] = 'None'
            break

    output.type.shape = output_shape


register_shape_calculator('flatten', calculate_flatten_output_shapes)
