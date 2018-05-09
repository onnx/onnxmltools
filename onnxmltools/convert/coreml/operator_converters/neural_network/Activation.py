# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from ....common._registration import register_converter


def convert_activation(scope, operator, container):
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

    params = operator.raw_operator.activation
    activation_type = params.WhichOneof('NonlinearityType')
    if activation_type == 'leakyReLU':
        op_type = 'LeakyRelu'
        attrs['alpha'] = params.leakyReLU.alpha
    elif activation_type == 'ReLU':
        op_type = 'Relu'
    elif activation_type == 'PReLU':
        op_type = 'PRelu'
        attrs['slope'] = params.PReLU.alpha
    elif activation_type == 'ELU':
        op_type = 'Elu'
        attrs['alpha'] = params.ELU.alpha
    elif activation_type == 'thresholdedReLU':
        op_type = 'ThresholdedRelu'
        attrs['alpha'] = params.thresholdedReLU.alpha
    elif activation_type == 'tanh':
        op_type = 'Tanh'
    elif activation_type == 'scaledTanh':
        op_type = 'ScaledTanh'
        attrs['alpha'] = params.scaledTanh.alpha
        attrs['beta'] = params.scaledTanh.beta
    elif activation_type == 'linear':
        op_type = 'Affine'
        attrs['alpha'] = params.linear.alpha
        attrs['beta'] = params.linear.beta
    elif activation_type == 'sigmoid':
        op_type = 'Sigmoid'
    elif activation_type == 'sigmoidHard':
        op_type = 'HardSigmoid'
        attrs['alpha'] = params.sigmoidHard.alpha
        attrs['beta'] = params.sigmoidHard.beta
    elif activation_type == 'softsign':
        op_type = 'Softsign'
    elif activation_type == 'softplus':
        op_type = 'Softplus'
    elif activation_type == 'parametricSoftplus':
        op_type = 'ParametricSoftplus'
        attrs['alpha'] = params.parametricSoftplus.alpha
        attrs['beta'] = params.parametricSoftplus.beta
    else:
        raise TypeError('Unsupported activation layer {0}'.format(activation_type))

    # For some ONNX versions, some activation operators include a legacy optimization attribute.
    op_version = 1
    if op_type in ['LeakyRelu', 'ReLU', 'PReLU', 'ELU', 'Tanh', 'Sigmoid', 'HardSigmoid']:
        if container.targeted_onnx_version < StrictVersion('1.0'):
            attrs['consumed_inputs'] = [0]
        else:
            op_version = 6

    container.add_node(op_type, inputs, outputs, op_version=op_version, **attrs)


register_converter('activation', convert_activation)
