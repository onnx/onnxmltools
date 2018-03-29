# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_flatten_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Flatten layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if type(input.type) != FloatTensorType or len(input.type.shape) not in [2, 4]:
        raise RuntimeError('Input must be 2-D or 4-D float tensor')

    input_shape = input.type.shape
    output_shape = [input_shape[0], 1, 1, 1]

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
