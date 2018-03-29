# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._data_types import FloatTensorType
from ....common._registration import register_shape_calculator

def calculate_reshape_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reshape operator has only one input and output')

    params = operator.raw_operator.reshape

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Only float tensors can be reshaped')

    output_shape = list(int(i) for i in params.targetShape)

    if len(output_shape) == 3:
        output_shape = [operator.inputs[0].type.shape[0]] + output_shape

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('reshape', calculate_reshape_output_shapes)
