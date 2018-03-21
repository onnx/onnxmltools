# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ..._data_types import FloatTensorType
from ...registration import register_shape_calculator


def calculate_split_output_shapes(operator):
    if len(operator.inputs) != 1:
        raise RuntimeError('Split has only one input')

    if len(operator.inputs) < 1:
        raise RuntimeError('Split should create at least one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    divided = output_shape[1] / operator.raw_operator.split.nOutputs
    if divided != int(divided):
        raise RuntimeError('Variable dimension along C-axis must be divisible by partition number')

    output_shape[1] = int(divided)

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('split', calculate_split_output_shapes)
