# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers.core import RepeatVector
from ...common._apply_operation import apply_reshape, apply_tile
from ...common._registration import register_converter


def convert_keras_repeat_vector(scope, operator, container):
    op = operator.raw_operator

    intermediate_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_reshaped')
    apply_reshape(scope, operator.inputs[0].full_name, intermediate_tensor_name, container,
                  desired_shape=[-1, 1, op.input_shape[1]])

    repeats = [1, int(op.n), 1]
    apply_tile(scope, intermediate_tensor_name, operator.outputs[0].full_name, container, repeats=repeats)


register_converter(RepeatVector, convert_keras_repeat_vector)
