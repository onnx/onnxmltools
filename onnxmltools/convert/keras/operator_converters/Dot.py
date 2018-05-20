# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Dot
from ...common._apply_operation import apply_mul, apply_normalization
from ...common._registration import register_converter


def convert_keras_dot(scope, operator, container):
    op = operator.raw_operator
    if (op.axes not in [-1, 1]):
        raise RuntimeError('Unsupported axes value for dot conversion: %s' % op.axes)
    if (len(op.input_shape[0]) > 2 or len(op.input_shape[1]) > 2):
        raise RuntimeError('Unsupported input shape for dot conversion')

    normalized_input_names = []
    if op.normalize:
        for tensor_name in operator.input_full_names:
            normalized_tensor_name = scope.get_unique_variable_name(tensor_name)
            apply_normalization(scope, tensor_name, normalized_tensor_name, container)
            normalized_input_names.append(normalized_tensor_name)
    else:
        normalized_input_names = operator.input_full_names

    intermediate_tensor_name = scope.get_unique_variable_name('elementwise_product')
    apply_mul(scope, normalized_input_names, intermediate_tensor_name, container)

    container.add_node('ReduceSum', intermediate_tensor_name, operator.output_full_names,
                       name=scope.get_unique_operator_name('ReduceSum'), axes=[1], keepdims=1)


register_converter(Dot, convert_keras_dot)
