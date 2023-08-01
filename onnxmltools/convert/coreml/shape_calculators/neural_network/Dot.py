# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_dot_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C], [N, C] ---> [N, 1]
        2. [N, C, 1, 1], [N, C, 1, 1] ---> [N, 1, 1, 1]
    """
    check_input_and_output_numbers(operator, input_count_range=2, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        raise RuntimeError("Input shapes must be identical")

    # Assume that inputs are [N, C]- or [N, C, 1, 1]-tensors
    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    output_shape[1] = 1
    operator.outputs[0].type.shape = output_shape


register_shape_calculator("dot", calculate_dot_output_shapes)
