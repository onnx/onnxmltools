# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_merge_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=[1, None], output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    # [TODO] Fix reduce-like shape inference. We now assume all inputs are 4-D.
    output_shape = [0, 0, 0, 0]
    for i in range(4):
        input_dims = [variable.type.shape[i] for variable in operator.inputs]
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
