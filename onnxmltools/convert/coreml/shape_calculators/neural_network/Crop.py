# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_crop_output_shapes(operator):
    if len(operator.inputs) > 2 or len(operator.outputs) != 1:
        raise RuntimeError('Invalid input or output numbers')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.crop
    if len(operator.inputs) == 1:
        if len(params.cropAmounts.borderAmounts) > 0:
            output_shape[2] -= params.cropAmounts.borderAmounts[0].startEdgeSize
            output_shape[2] -= params.cropAmounts.borderAmounts[0].endEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].startEdgeSize
            output_shape[3] -= params.cropAmounts.borderAmounts[1].endEdgeSize
    elif len(operator.inputs) == 2:
        output_shape[2] = operator.raw_operator.inputs[1].type.shape[2]
        output_shape[3] = operator.raw_operator.inputs[1].type.shape[3]
    else:
        raise RuntimeError('Too many inputs for Crop operator')

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('crop', calculate_crop_output_shapes)
