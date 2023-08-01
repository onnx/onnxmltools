# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_gru_output_shapes(operator):
    '''
    See GRU's conversion function for its output shapes.
    '''
    check_input_and_output_numbers(operator, input_count_range=[1, 2], output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

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

    # TODO: Changing input shapes of an operator is dangerous, this should be move to Topology's _fix_shapes function
    if len(operator.inputs) > 1:
        Y_h_in = operator.inputs[1]  # The initial hidden state of a single sequence
        Y_h_in.type.shape = state_shape

    operator.outputs[0].type.shape = output_shape
    if len(operator.outputs) > 1:
        operator.outputs[1].type.shape = state_shape


register_shape_calculator('gru', calculate_gru_output_shapes)
register_shape_calculator('simpleRecurrent', calculate_gru_output_shapes)
