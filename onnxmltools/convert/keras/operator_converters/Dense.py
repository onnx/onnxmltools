# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.activations import get as _get_activation
from keras.layers import Dense
from ....proto import onnx_proto
from ...common._registration import register_converter

_activation_map = {_get_activation('sigmoid'): 'Sigmoid',
                   _get_activation('softmax'): 'Softmax',
                   _get_activation('linear'): 'Affine',
                   _get_activation('relu'): 'Relu'}


def convert_keras_dense(scope, operator, container):
    op_type = 'FC'
    attrs = {'name': operator.full_name}
    attrs['axis'] = 1
    attrs['axis_w'] = 1

    parameters = operator.raw_operator.get_weights()

    weight_name = scope.get_unique_variable_name('W')
    weight = parameters[0].transpose()
    container.add_initializer(weight_name, onnx_proto.TensorProto.FLOAT, weight.shape, weight.flatten())

    bias = parameters[1]
    bias_name = scope.get_unique_variable_name('B')
    container.add_initializer(bias_name, onnx_proto.TensorProto.FLOAT, bias.shape, bias.flatten())

    # Create a fully-connected layer and put the output on an intermediate tensor
    intermediate_tensor_name = scope.get_unique_variable_name('intermediate')
    container.add_node(op_type, [operator.inputs[0].full_name, weight_name, bias_name],
                       intermediate_tensor_name, **attrs)

    # Create an activation function node and apply activation function to the intermediate tensor
    activation_type = _activation_map[operator.raw_operator.activation]
    activation_attrs = {'name': scope.get_unique_operator_name(activation_type)}
    container.add_node(activation_type, intermediate_tensor_name, operator.outputs[0].full_name, **activation_attrs)


register_converter(Dense, convert_keras_dense)
