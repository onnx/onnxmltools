# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_lstm_output_shapes(operator):
    for variable in operator.inputs:
        if type(variable.type) != FloatTensorType:
            raise RuntimeError('LSTM only accepts float tensors as inputs')

    input_shape = operator.inputs[0].type.shape

    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D tensor')

    params = operator.raw_operator.uniDirectionalLSTM

    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [1, params.outputVectorSize]
    output_shape = ['None', params.outputVectorSize]
    state_shape = [1, params.outputVectorSize]

    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The initial cell state of a single sequence
        Y_c_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape
    if len(operator.outputs) > 2:
        operator.outputs[2].type.shape = state_shape


register_shape_calculator('uniDirectionalLSTM', calculate_lstm_output_shapes)
