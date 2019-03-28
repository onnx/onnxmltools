#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types

def calculate_inner_product_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C] ---> [N, C']
        2. [N, C, 1, 1] ---> [N, C', 1, 1]
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input = operator.inputs[0]
    output = operator.outputs[0]

    input_shape = input.type.shape
    if len(input_shape) == 4 and (input_shape[2] != 1 or input_shape[3] != 1):
        raise RuntimeError('If input is a 4-D tensor, its shape must be [N, C, 1, 1]')

    params = operator.raw_operator.innerProduct

    if input_shape[1] != params.inputChannels:
        raise RuntimeError('Dimension mismatch along C-axis. Expected %s but got %s' %
                           (params.inputChannels, input_shape[1]))

    if len(input_shape) == 4:
        output.type.shape = [input_shape[0], params.outputChannels, 1, 1]
    elif len(input_shape) == 2:
        output.type.shape = [input_shape[0], params.outputChannels]
    else:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')


register_shape_calculator('innerProduct', calculate_inner_product_output_shapes)
