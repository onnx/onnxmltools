# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_merge_output_shapes(operator):
    if len(operator.inputs) < 1:
        raise RuntimeError('Add operator requires at least one input')
    if len(operator.outputs) != 1:
        raise RuntimeError('Add operator only has one output')

    for variable in operator.inputs:
        if not isinstance(variable.type, FloatTensorType):
            raise RuntimeError('Input must be a float tensor')

    # [TODO] Fix reduce-like shape inference. We now assume all inputs are 4-D.
    output_shape = [0, 0, 0, 0]
    for i in range(4):
        input_dims = [variable.type.shape[i] for variable in operator.inputs]
        if 'None' in input_dims:
            output_shape[i] = 'None'
        else:
            output_shape[i] = max(input_dims)

    operator.outputs[0].type.shape = output_shape


register_shape_calculator('add', calculate_merge_output_shapes)
register_shape_calculator('average', calculate_merge_output_shapes)
register_shape_calculator('max', calculate_merge_output_shapes)
register_shape_calculator('min', calculate_merge_output_shapes)
register_shape_calculator('multiply', calculate_merge_output_shapes)
