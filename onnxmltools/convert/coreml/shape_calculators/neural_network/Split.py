# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_split_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N', C]
        2. [N, C, H, W] ---> [N', C, H, W]
    """
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, None]
    )
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    divided = output_shape[1] / operator.raw_operator.split.nOutputs
    if divided != int(divided):
        raise RuntimeError(
            "Variable dimension along C-axis must be divisible by partition number"
        )

    output_shape[1] = int(divided)

    for i in range(len(operator.outputs)):
        operator.outputs[i].type.shape = copy.deepcopy(output_shape)


register_shape_calculator("split", calculate_split_output_shapes)
