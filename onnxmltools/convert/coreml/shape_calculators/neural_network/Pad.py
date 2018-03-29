# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_padding_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Padding is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.padding
    if len(params.paddingAmounts.borderAmounts) > 0:
        output_shape[2] += params.paddingAmounts.borderAmounts[0].startEdgeSize
        output_shape[2] += params.paddingAmounts.borderAmounts[0].endEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].startEdgeSize
        output_shape[3] += params.paddingAmounts.borderAmounts[1].endEdgeSize

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('padding', calculate_padding_output_shapes)
