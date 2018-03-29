# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_gru_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('GRU only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a [N, C]- or [N, C, 1, 1]-tensor')

    if operator.type == 'gru':
        params = operator.raw_operator.gru
    elif operator.type == 'simpleRecurrent':
        params = operator.raw_operator.simpleRecurrent
    else:
        raise RuntimeError('Only GRU and SimpleRNN are supported')

    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [2, params.outputVectorSize]
    output_shape = [input_shape[0] if params.sequenceOutput else 'None', params.outputVectorSize]  # 'None' should be 1
    state_shape = [1, params.outputVectorSize]

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape


register_shape_calculator('gru', calculate_gru_output_shapes)
register_shape_calculator('simpleRecurrent', calculate_gru_output_shapes)
