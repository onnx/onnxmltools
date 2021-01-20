# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_shape_calculator
from ....common.data_types import Int64Type, Int64TensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_embedding_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, 1] ---> [N, C]
        2. [N, 1, 1, 1] ---> [N, C, 1, 1]
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[Int64Type, Int64TensorType])

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
