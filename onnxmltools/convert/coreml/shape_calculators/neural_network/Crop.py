# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import (
    check_input_and_output_numbers,
    check_input_and_output_types,
)


def calculate_crop_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C, H', W']
        2. [N, C, H, W],  shape-ref [N', C', H', W'] ---> [N, C, H', W']
    """
    check_input_and_output_numbers(
        operator, input_count_range=[1, 2], output_count_range=1
    )
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.crop
    if len(operator.inputs) == 1:
        if len(params.cropAmounts.borderAmounts) > 0:
            output_shape[2] -= params.cropAmounts.borderAmounts[0].startEdgeSize
            output_shape[2] -= params.cropAmounts.borderAmounts[0].endEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].startEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].endEdgeSize
    elif len(operator.inputs) == 2:
        output_shape[2] = operator.inputs[1].type.shape[2]
        output_shape[3] = operator.inputs[1].type.shape[3]
    else:
        raise RuntimeError("Too many inputs for Crop operator")

    operator.outputs[0].type.shape = output_shape


register_shape_calculator("crop", calculate_crop_output_shapes)
