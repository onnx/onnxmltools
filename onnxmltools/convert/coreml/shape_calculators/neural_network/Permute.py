# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._data_types import TensorType
from ....common._registration import register_shape_calculator


def calculate_permute_output_shapes(operator):
    if len(operator.inputs) > 1 or len(operator.outputs) > 1:
        raise RuntimeError('Permute layer can only have one input and one output')

    input = operator.inputs[0]
    output = operator.outputs[0]

    if not isinstance(input.type, TensorType) or not isinstance(output.type, TensorType):
        raise RuntimeError('Only tensor types can be permuted')

    axes = [int(i) for i in operator.raw_operator.permute.axis]
    input_shape = copy.deepcopy(input.type.shape)
    output.type.shape = [input_shape[a] for a in axes]


register_shape_calculator('permute', calculate_permute_output_shapes)
