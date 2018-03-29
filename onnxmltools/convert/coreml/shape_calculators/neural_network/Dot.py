# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_dot_output_shapes(operator):
    if len(operator.inputs) != 2 or len(operator.outputs) != 1:
        raise RuntimeError('Dot must have two inputs and one output')
    if any(type(variable.type) != FloatTensorType for variable in operator.inputs):
        raise RuntimeError('Input(s) must be float tensor(s)')
    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        raise RuntimeError('Input shapes must be identical')

    # Assume that inputs are [N, C]- or [N, C, 1, 1]-tensors
    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    output_shape[1] = 1
    operator.outputs[0].type.shape = output_shape


register_shape_calculator('dot', calculate_dot_output_shapes)
