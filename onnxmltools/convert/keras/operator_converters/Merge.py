# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras.layers
from ...common._registration import register_converter

_merge_layer_type_map = {keras.layers.Add: 'Add', keras.layers.Multiply: 'Mul', keras.layers.Subtract: 'Sub',
                         keras.layers.Average: 'Mean', keras.layers.Maximum: 'Max'}


def convert_keras_merge_layer(scope, operator, container):
    op = operator.raw_operator
    if isinstance(op, keras.layers.Subtract) and len(operator.inputs) > 2:
        raise RuntimeError(
            'Expected two inputs but got %s. Their names are %s' % (len(operator.inputs), operator.input_full_names))

    op_type = _merge_layer_type_map[type(op)]

    intermediate_tensor_name = None
    for i in range(len(operator.inputs) - 1):
        if i == 0:
            left_tensor_name = operator.inputs[0].full_name
            right_tensor_name = operator.inputs[1].full_name
            op_name = operator.full_name
        else:
            if intermediate_tensor_name is None:
                raise RuntimeError('Tensor name cannot be None')
            left_tensor_name = intermediate_tensor_name
            right_tensor_name = operator.inputs[i + 1].full_name
            op_name = scope.get_unique_operator_name(op_type)

        if (len(operator.inputs) == 2 and i == 0) or (len(operator.inputs) > 2 and i == len(operator.inputs) - 2):
            # At the last iteration, we need to put the result to Keras layer's output tensor
            intermediate_tensor_name = operator.outputs[0].full_name
        else:
            # Keep accumulate changes through iterations using buffer tensors
            intermediate_tensor_name = scope.get_unique_variable_name('intermediate_tensor')
        container.add_node(op_type, [left_tensor_name, right_tensor_name], intermediate_tensor_name,
                           name=op_name)


register_converter(keras.layers.Add, convert_keras_merge_layer)
register_converter(keras.layers.Multiply, convert_keras_merge_layer)
register_converter(keras.layers.Subtract, convert_keras_merge_layer)
register_converter(keras.layers.Average, convert_keras_merge_layer)
register_converter(keras.layers.Maximum, convert_keras_merge_layer)
