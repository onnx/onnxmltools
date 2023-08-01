# SPDX-License-Identifier: Apache-2.0

import copy
from ....common.utils import check_input_and_output_numbers
from ....common._registration import register_shape_calculator


def calculate_concat_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N_1, C, H, W], ..., [N_n, C, H, W] ---> [N_1 + ... + N_n, C, H, W]
        2. [N, C_1, H, W], ..., [N, C_n, H, W] ---> [N, C_1 + ... + C_n, H, W]
    """
    check_input_and_output_numbers(
        operator, input_count_range=[1, None], output_count_range=[1, 1]
    )

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    dims = []
    for variable in operator.inputs:
        if (
            variable.type.shape[0] != "None"
            and variable.type.shape[0] != output_shape[0]
        ):
            raise RuntimeError("Only dimensions along C-axis can be different")
        if (
            variable.type.shape[2] != "None"
            and variable.type.shape[2] != output_shape[2]
        ):
            raise RuntimeError("Only dimensions along C-axis can be different")
        if (
            variable.type.shape[3] != "None"
            and variable.type.shape[3] != output_shape[3]
        ):
            raise RuntimeError("Only dimensions along C-axis can be different")
        dims.append(variable.type.shape[1])

    output_shape[1] = "None" if "None" in dims else sum(dims)
    operator.outputs[0].type.shape = output_shape


register_shape_calculator("concat", calculate_concat_output_shapes)
