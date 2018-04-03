# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras.layers.advanced_activations as activations
from ...common._registration import register_converter
from ....proto import onnx_proto


def convert_keras_advanced_activation(scope, operator, container):
    op = operator.raw_operator
    attrs = {'name': operator.full_name}
    input_tensor_names = operator.input_full_names
    if isinstance(op, activations.LeakyReLU):
        op_type = 'LeakyRelu'
        attrs['alpha'] = op.get_config()['alpha']
    elif isinstance(op, activations.ELU):
        op_type = 'Elu'
        attrs['alpha'] = op.get_config()['alpha']
    elif isinstance(op, activations.ThresholdedReLU):
        op_type = 'ThresholdedRelu'
        attrs['alpha'] = op.get_config()['theta']
    elif isinstance(op, activations.PReLU):
        op_type = 'PRelu'
        parameters = op.get_weights()
        slope_tensor_name = scope.get_unique_variable_name('slope')
        container.add_initializer(slope_tensor_name, onnx_proto.TensorProto.FLOAT,
                                  parameters[0].shape, parameters[0].flatten())
        input_tensor_names.append(slope_tensor_name)
    # TODO:Following layer is not supported by the checked-in keras version and requires an upgrade of the checked-in keras
    # elif isinstance(op, activations.Softmax):
    #    attrs['axis'] = op.get_config()['axis']
    else:
        raise RuntimeError('Unsupported advanced layer found %s' % type(op))

    container.add_node(op_type, input_tensor_names, operator.output_full_names, **attrs)


register_converter(activations.LeakyReLU, convert_keras_advanced_activation)
register_converter(activations.ThresholdedReLU, convert_keras_advanced_activation)
register_converter(activations.ELU, convert_keras_advanced_activation)
register_converter(activations.PReLU, convert_keras_advanced_activation)
# TODO:Following layer is not supported by the checked-in keras version and requires an upgrade of the checked-in keras
# register_converter(activations.Softmax, convert_keras_advanced_activation)
