# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common._data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_batch_normalization_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Batch normalization is an one-to-one mapping')

    if not isinstance(operator.inputs[0].type, FloatTensorType):
        raise RuntimeError('Input must be a float tensor')

    input_shape = operator.inputs[0].type.shape
    if len(input_shape) not in [2, 4]:
        raise RuntimeError('Input must be a 2-D or a 4-D tensor')

    operator.outputs[0].type.shape = copy.deepcopy(operator.inputs[0].type.shape)


register_shape_calculator('batchnorm', calculate_batch_normalization_output_shapes)
