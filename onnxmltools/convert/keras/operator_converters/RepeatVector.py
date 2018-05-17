# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers.core import RepeatVector
from ...common._registration import register_converter


def convert_keras_repeat_vector(scope, operator, container):
    op = operator.raw_operator
    intermediate_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_reshaped')

    container.add_node('Reshape', operator.inputs[0].full_name, intermediate_tensor_name,
                       name=scope.get_unique_operator_name('Reshape'), shape=[-1, 1, op.input_shape[1]])

    container.add_node('Tile', intermediate_tensor_name, operator.outputs[0].full_name,
                       name=operator.full_name, tiles=op.n, axis=1)


register_converter(RepeatVector, convert_keras_repeat_vector)
