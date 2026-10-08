# SPDX-License-Identifier: Apache-2.0

import copy

from ...common._registration import register_shape_calculator
from ...common.shape_calculator import check_input_and_output_numbers


def calculate_cast_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    operator.outputs[0].type.shape = copy.deepcopy(operator.inputs[0].type.shape)


register_shape_calculator("cast", calculate_cast_output_shapes)
