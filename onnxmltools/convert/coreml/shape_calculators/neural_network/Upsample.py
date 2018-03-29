# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_upsample_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Upsample has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')
    scales = operator.raw_operator.upsample.scalingFactor

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)
    output_shape[2] *= scales[0]
    output_shape[3] *= scales[1]

    operator.outputs[0].type = FloatTensorType(output_shape, doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('upsample', calculate_upsample_output_shapes)
