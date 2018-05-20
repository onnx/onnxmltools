# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras.layers
from ...common._apply_operation import apply_add, apply_mul, apply_sub, apply_mean, apply_max
from ...common._registration import register_converter

_merge_layer_handlers = {keras.layers.Add: apply_add, keras.layers.Multiply: apply_mul,
                         keras.layers.Subtract: apply_sub, keras.layers.Average: apply_mean,
                         keras.layers.Maximum: apply_max}


def convert_keras_merge_layer(scope, operator, container):
    op = operator.raw_operator
    if isinstance(op, keras.layers.Subtract) and len(operator.inputs) > 2:
        raise RuntimeError(
            'Expected two inputs but got %s. Their names are %s' % (len(operator.inputs), operator.input_full_names))

    apply_merge_operation = _merge_layer_handlers[type(op)]

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
            op_name = scope.get_unique_operator_name('Merge')

        if (len(operator.inputs) == 2 and i == 0) or (len(operator.inputs) > 2 and i == len(operator.inputs) - 2):
            # At the last iteration, we need to put the result to Keras layer's output tensor
            intermediate_tensor_name = operator.outputs[0].full_name
        else:
            # Keep accumulate changes through iterations using buffer tensors
            intermediate_tensor_name = scope.get_unique_variable_name('intermediate_tensor')
        apply_merge_operation(scope, [left_tensor_name, right_tensor_name], intermediate_tensor_name, container)


register_converter(keras.layers.Add, convert_keras_merge_layer)
register_converter(keras.layers.Multiply, convert_keras_merge_layer)
register_converter(keras.layers.Subtract, convert_keras_merge_layer)
register_converter(keras.layers.Average, convert_keras_merge_layer)
register_converter(keras.layers.Maximum, convert_keras_merge_layer)
