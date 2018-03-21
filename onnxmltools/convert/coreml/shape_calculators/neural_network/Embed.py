# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..._data_types import Int64Type, Int64TensorType
from ...registration import register_shape_calculator


def calculate_embedding_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Embedding layer can only have one input and one output')

    if type(operator.inputs[0].type) not in [Int64Type, Int64TensorType]:
        raise RuntimeError('ONNX embedding only accepts integer input')

    output = operator.outputs[0]

    input_shape = operator.inputs[0].type.shape

    if input_shape[1] != 1 or (len(input_shape) > 2 and (input_shape[2] != 1 or input_shape[3] != 1)):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, 1, 1, 1]')

    params = operator.raw_operator.embedding
    if len(input_shape) == 4:
        output_shape = [input_shape[0], params.outputChannels, 1, 1]
    elif len(input_shape) == 2:
        output_shape = [input_shape[0], params.outputChannels]
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    output.type.shape = output_shape


register_shape_calculator('embedding', calculate_embedding_output_shapes)
