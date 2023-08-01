# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_bidirectional_lstm_output_shapes(operator):
    '''
    See bidirectional LSTM's conversion function for its output shapes.
    '''
    check_input_and_output_numbers(operator, input_count_range=[1, 5], output_count_range=[1, 5])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input_shape = operator.inputs[0].type.shape

    # LSTM accepts [N, C] and [N, C, 1, 1] inputs
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or 4-D tensor')

    params = operator.raw_operator.biDirectionalLSTM
    # The following line is more accurate but it may break some tests
    # output_shape = ['None', params.outputVectorSize] if params.params.sequenceOutput else [1, 2 *params.outputVectorSize]
    output_shape = ['None', 2 * params.outputVectorSize]
    state_shape = [1, params.outputVectorSize]

    # TODO: Changing input shapes of an operator is dangerous, this should be move to Topology's _fix_shapes function
    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The forward initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape
        Y_h_rev_in = operator.inputs[3]  # The backward initial hidden state of a single sequence
        Y_h_rev_in.type.shape = state_shape
    if len(operator.inputs) > 2:
        Y_c_in = operator.inputs[2]  # The forward initial cell state of a single sequence
        Y_c_in.type.shape = state_shape
        Y_c_rev_in = operator.inputs[4]  # The backward initial cell state of a single sequence
        Y_c_rev_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape
        operator.outputs[3].type.shape = state_shape
    if len(operator.outputs) > 2:
        operator.outputs[2].type.shape = state_shape
        operator.outputs[4].type.shape = state_shape


register_shape_calculator('biDirectionalLSTM', calculate_bidirectional_lstm_output_shapes)
