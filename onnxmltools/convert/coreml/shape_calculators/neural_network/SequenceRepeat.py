# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_sequence_repeat_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Sequence Repeqt has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    if output_shape[0] != None:
        output_shape[0] *= operator.raw_operator.sequenceRepeat.nRepetitions

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('sequenceRepeat', calculate_sequence_repeat_output_shapes)
