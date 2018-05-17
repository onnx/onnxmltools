# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from keras.activations import get as _get_activation
from ...common._registration import register_converter


def activation_helper(activation):
    activation_attributes = {}
    if activation in [_get_activation('sigmoid'), keras.activations.sigmoid]:
        activation_type = 'Sigmoid'
    elif activation in [_get_activation('tanh'), keras.activations.tanh]:
        activation_type = 'Tanh'
    elif activation in [_get_activation('relu'), keras.activations.relu]:
        activation_type = 'Relu'
    elif activation in [_get_activation('softsign'), keras.activations.softsign]:
        activation_type = 'Softsign'
    elif activation in [_get_activation('softplus'), keras.activations.softplus]:
        activation_type = 'Softplus'
    elif activation in [_get_activation('softmax'), keras.activations.softmax]:
        activation_type = 'Softmax'
        activation_attributes = {'axis': -1}
    elif activation in [_get_activation('elu'), keras.activations.elu]:
        activation_type = 'Elu'
        activation_attributes = {'alpha': 1.0}
    elif activation in [_get_activation('hard_sigmoid'), keras.activations.hard_sigmoid]:
        activation_type = 'HardSigmoid'
        activation_attributes = {'alpha': 0.2, 'beta': 0.5}
    elif activation in [_get_activation('linear'), keras.activations.linear]:
        activation_type = 'Identity'
    elif activation in [_get_activation('selu'), keras.activations.selu]:
        activation_type = 'Selu'
        activation_attributes = {'alpha': 1.673263, 'gamma': 1.050700}
    else:
        raise RuntimeError('Unsupported activation method within Activation layer {}'.format(activation))

    return activation_type, activation_attributes


def convert_keras_activation(scope, operator, container):
    op_type, attrs = activation_helper(operator.raw_operator.activation)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=operator.full_name, **attrs)


register_converter(keras.layers.Activation, convert_keras_activation)
