# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_sequence_repeat_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N', C, H, W]
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

    input_shape = operator.inputs[0].type.shape
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input shape of CoreML SequenceRepeat must be either 2-D or 4-D but got %s' % input_shape)

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    if output_shape[0] != None:
        output_shape[0] *= operator.raw_operator.sequenceRepeat.nRepetitions

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('sequenceRepeat', calculate_sequence_repeat_output_shapes)
