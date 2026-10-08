# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_reshape_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C', H', W']

    Note that C*H*W should equal to C'*H'*W'.
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    params = operator.raw_operator.reshape

    output_shape = list(int(i) for i in params.targetShape)

    if len(output_shape) == 3:
        output_shape = [operator.inputs[0].type.shape[0]] + output_shape

    operator.outputs[0].type.shape = output_shape


register_shape_calculator("reshape", calculate_reshape_output_shapes)
