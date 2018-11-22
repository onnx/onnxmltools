# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
import keras.layers.advanced_activations as activations
from distutils.version import StrictVersion
from ...common._apply_operation import apply_elu, apply_leaky_relu, apply_prelu
from ...common._registration import register_converter


def convert_keras_advanced_activation(scope, operator, container):
    op = operator.raw_operator
    if isinstance(op, activations.LeakyReLU):
        alpha = op.get_config()['alpha']
        apply_leaky_relu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                         operator_name=operator.full_name, alpha=alpha)
    elif isinstance(op, activations.ELU):
        alpha = op.get_config()['alpha']
        apply_elu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                  operator_name=operator.full_name, alpha=alpha)
    elif isinstance(op, activations.PReLU):
        weights = op.get_weights()[0]
        apply_prelu(scope, operator.input_full_names[0], operator.output_full_names[0], container,
                    operator_name=operator.full_name, slope=weights)
    else:
        attrs = {'name': operator.full_name}
        ver_opset = 6
        input_tensor_names = [operator.input_full_names[0]]
        if isinstance(op, activations.ThresholdedReLU):
            op_type = 'ThresholdedRelu'
            attrs['alpha'] = op.get_config()['theta']
        elif StrictVersion(keras.__version__) >= StrictVersion('2.1.3') and \
                isinstance(op, activations.Softmax):
            op_type = 'Softmax'
            attrs['axis'] = op.get_config()['axis']
        elif StrictVersion(keras.__version__) >= StrictVersion('2.2.0') and \
                isinstance(op, activations.ReLU):
            if op.threshold != 0.0:
                raise RuntimeError('Threshold in ReLU is not supported in ONNX')
            op_type = 'Relu'
        else:
            raise RuntimeError('Unsupported advanced layer found %s' % type(op))

        container.add_node(op_type, input_tensor_names, operator.output_full_names, op_version=ver_opset, **attrs)


register_converter(activations.LeakyReLU, convert_keras_advanced_activation)
register_converter(activations.ThresholdedReLU, convert_keras_advanced_activation)
register_converter(activations.ELU, convert_keras_advanced_activation)
register_converter(activations.PReLU, convert_keras_advanced_activation)
if StrictVersion(keras.__version__) >= StrictVersion('2.1.3'):
    register_converter(activations.Softmax, convert_keras_advanced_activation)
if StrictVersion(keras.__version__) >= StrictVersion('2.2.0'):
    register_converter(activations.ReLU, convert_keras_advanced_activation)
