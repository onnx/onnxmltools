#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....common._data_types import TensorType, FloatTensorType
from ....common._registration import register_shape_calculator

def calculate_load_constant_output_shapes(operator):
    if len(operator.inputs) != 0:
        raise RuntimeError('Load Constant operator has no input')
    if len(operator.outputs) != 1:
        raise RuntimeError('Load Constant operator has only one output')
    output = operator.outputs[0]

    # CoreML's constant is always 3-D tensor, so we assume its shape is [C, H, W].
    const_shape = operator.raw_operator.loadConstant.shape
    # We convert [C, H, W] to [1, C, H, W] because our parsing code use [N, C, H, W]
    const_shape = [1] + [int(d) for d in const_shape]
    if output.type is None:
        # Use default type
        output.type = FloatTensorType(const_shape, doc_string=output.type.doc_string)
    else:
        if not isinstance(output.type, TensorType):
            raise RuntimeError('Type conflict detected. Output must be a tensor.')
        # If output type exists, we just modify its shape.
        output.type.shape = const_shape


register_shape_calculator('loadConstant', calculate_load_constant_output_shapes)
