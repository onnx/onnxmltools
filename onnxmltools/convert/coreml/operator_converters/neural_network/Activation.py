# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_activation(scope, operator, container):
    params = operator.raw_operator.activation
    activation_type = params.WhichOneof('NonlinearityType')

    # Compose special operators with ONNX primitives. Those functions are not directly defined in ONNX.
    if activation_type == 'scaledTanh':
        alpha_tensor_name = scope.get_unique_variable_name(operator.full_name + '_alpha')
        beta_name = scope.get_unique_variable_name(operator.full_name + '_beta')
        container.add_initializer(alpha_tensor_name, onnx_proto.TensorProto.FLOAT, [], [params.scaledTanh.alpha])
        container.add_initializer(beta_name, onnx_proto.TensorProto.FLOAT, [], [params.scaledTanh.beta])

        intra_tensor_name1 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_scaled')
        container.add_node('Mul', [operator.inputs[0].full_name, beta_name], intra_tensor_name1,
                           name=scope.get_unique_operator_name('Mul'), broadcast=1)
        intra_tensor_name2 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_tanh')
        container.add_node('Tanh', intra_tensor_name1, intra_tensor_name2,
                           name=operator.full_name)
        container.add_node('Mul', [intra_tensor_name2, alpha_tensor_name], operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Mul'), broadcast=1)
        return
    elif activation_type == 'thresholdedReLU':
        intra_tensor_name1 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_mask')
        intra_tensor_name2 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_mask_casted')
        alpha_tensor_name = scope.get_unique_variable_name(operator.full_name + '_alpha')
        container.add_initializer(alpha_tensor_name, onnx_proto.TensorProto.FLOAT, [], [params.thresholdedReLU.alpha])

        container.add_node('Greater', [operator.inputs[0].full_name, alpha_tensor_name], intra_tensor_name1,
                           name=scope.get_unique_operator_name('Less'), broadcast=1)

        container.add_node('Cast', [operator.inputs[0].full_name, alpha_tensor_name], intra_tensor_name2,
                           name=scope.get_unique_operator_name('Cast'), to=onnx_proto.TensorProto.FLOAT)

        container.add_node('Mul', [operator.inputs[0].full_name, intra_tensor_name2], operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Mul'), broadcast=0)
        return
    elif activation_type == 'parametricSoftplus':
        alpha_value = params.parametricSoftplus.alpha
        beta_value = params.parametricSoftplus.beta
        if len(alpha_value) == 1:
            alpha_broadcast_axis = None
            alpha_value = [float(alpha_value)]
            alpha_shape = []  # Scalar shape
        else:
            alpha_broadcast_axis = 1  # along C-axi because alpha is a [C]-element vector
            alpha_shape = [len(alpha_value)]
        if len(beta_value) == 1:
            beta_broadcast_axis = None
            beta_value = [float(beta_value)]
            beta_shape = []  # Scalar shape
        else:
            beta_broadcast_axis = 1  # along C-axis because beta is a [C]-element vector
            beta_shape = [len(beta_value)]

        alpha_tensor_name = scope.get_unique_variable_name(operator.full_name + '_alpha')
        beta_tensor_name = scope.get_unique_variable_name(operator.full_name + '_beta')
        one_tensor_name = scope.get_unique_variable_name(operator.full_name + '_one')
        container.add_initializer(alpha_tensor_name, onnx_proto.TensorProto.FLOAT, alpha_shape, alpha_value)
        container.add_initializer(beta_tensor_name, onnx_proto.TensorProto.FLOAT, beta_shape, beta_value)
        container.add_initializer(one_tensor_name, onnx_proto.TensorProto.FLOAT, [], [1.])

        intra_tensor_name1 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_beta_scaled')
        intra_tensor_name2 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_exp')
        intra_tensor_name3 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_one_added')
        intra_tensor_name4 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_log')

        if alpha_broadcast_axis is None:
            container.add_node('Mul', [operator.inputs[0].full_name, beta_tensor_name], intra_tensor_name1,
                               name=scope.get_unique_operator_name('Mul'), broadcast=1)
        else:
            container.add_node('Mul', [operator.inputs[0].full_name, beta_tensor_name], intra_tensor_name1,
                               name=scope.get_unique_operator_name('Mul'), broadcast=1, axis=alpha_broadcast_axis)

        container.add_node('Exp', intra_tensor_name1, intra_tensor_name2,
                           name=scope.get_unique_operator_name('Exp'))

        container.add_node('Add', [intra_tensor_name2, one_tensor_name], intra_tensor_name3,
                           name=scope.get_unique_operator_name('Add'), broadcast=1)

        container.add_node('Log', intra_tensor_name3, intra_tensor_name4,
                           name=scope.get_unique_operator_name('Log'))

        if beta_broadcast_axis is None:
            container.add_node('Mul', [intra_tensor_name4, alpha_tensor_name], operator.outputs[0].full_name,
                               name=scope.get_unique_operator_name('Mul'), broadcast=1)
        else:
            container.add_node('Mul', [intra_tensor_name4, alpha_tensor_name], operator.outputs[0].full_name,
                               name=scope.get_unique_operator_name('Mul'), broadcast=1, axis=beta_broadcast_axis)
        return

    # The following operators are natively supported in ONNX
    inputs = [variable.full_name for variable in operator.inputs]
    outputs = [variable.full_name for variable in operator.outputs]
    attrs = {'name': operator.full_name}

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
    elif activation_type == 'tanh':
        op_type = 'Tanh'
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
    else:
        raise TypeError('Unsupported activation layer {0}'.format(activation_type))

    container.add_node(op_type, inputs, outputs, **attrs)


register_converter('activation', convert_activation)
