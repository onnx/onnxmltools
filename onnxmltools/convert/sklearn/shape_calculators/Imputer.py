# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common.data_types import FloatTensorType, Int64TensorType
from ...common._registration import register_shape_calculator


def calculate_sklearn_imputer_output_shapes(operator):
    if len(operator.inputs) < 1 or len(operator.outputs) != 1:
        raise RuntimeError('Invalid input or output numbers')
    if any(not isinstance(variable.type, (FloatTensorType, Int64TensorType)) for variable in operator.inputs):
        raise RuntimeError('Input(s) must be integer- or float-tensor(s)')
    C = 0
    for variable in operator.inputs:
        if variable.type.shape[1] != 'None':
            C += variable.type.shape[1]
        else:
            C = 'None'
            break

    operator.outputs[0].type = FloatTensorType([1, C])


register_shape_calculator('SklearnImputer', calculate_sklearn_imputer_output_shapes)
