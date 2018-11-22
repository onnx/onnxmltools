# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import warnings
import numpy
import keras
from keras.activations import get as _get_activation
from ...common._apply_operation import apply_elu, apply_hard_sigmoid, apply_relu, apply_sigmoid, apply_tanh, \
    apply_softmax, apply_identity, apply_selu, apply_clip
from ...common._registration import register_converter

try:
    from keras_applications.mobilenet import relu6
except ImportError:
    relu6 = None


def convert_keras_activation(scope, operator, container):
    input_name = operator.input_full_names[0]
    output_name = operator.output_full_names[0]
    activation = operator.raw_operator.activation
    activation_attributes = {}
    if activation in [_get_activation('sigmoid'), keras.activations.sigmoid]:
        apply_sigmoid(scope, input_name, output_name, container)
    elif activation in [_get_activation('tanh'), keras.activations.tanh]:
        apply_tanh(scope, input_name, output_name, container)
    elif activation in [_get_activation('relu'), keras.activations.relu]:
        apply_relu(scope, input_name, output_name, container)
    elif activation in [_get_activation('softmax'), keras.activations.softmax]:
        apply_softmax(scope, input_name, output_name, container, axis=-1)
    elif activation in [_get_activation('elu'), keras.activations.elu]:
        apply_elu(scope, input_name, output_name, container, alpha=1.0)
    elif activation in [_get_activation('hard_sigmoid'), keras.activations.hard_sigmoid]:
        apply_hard_sigmoid(scope, input_name, output_name, container, alpha=0.2, beta=0.5)
    elif activation in [_get_activation('linear'), keras.activations.linear]:
        apply_identity(scope, input_name, output_name, container)
    elif activation in [_get_activation('selu'), keras.activations.selu]:
        apply_selu(scope, input_name, output_name, container, alpha=1.673263, gamma=1.050700)
    elif activation in [relu6]:
        # relu6(x) = min(relu(x), 6)
        apply_relu(scope, input_name, output_name + "_relu6", container)
        apply_clip(scope, output_name + "_relu6", output_name, container,
                   min=0, max=6)
    else:
        if activation in [_get_activation('softsign'), keras.activations.softsign]:
            op_type = 'Softsign'
        elif activation in [_get_activation('softplus'), keras.activations.softplus]:
            op_type = 'Softplus'
        else:
            raise RuntimeError("Unsupported activation method within Activation layer '{}'".format(activation))

        container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=operator.full_name)


register_converter(keras.layers.Activation, convert_keras_activation)
