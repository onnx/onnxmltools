# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_unary(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UnaryFunctionLayerParams as Params

    params = operator.raw_operator.unary

    alpha, beta = params.scale, params.shift

    # Declare intermediate tensor names. They will be used to build the preprocessing step before feeding the input into
    # subsequent operators.
    alpha_tensor_name = scope.get_unique_variable_name(operator.full_name + '_scale')
    beta_tensor_name = scope.get_unique_variable_name(operator.full_name + '_shift')
    scaled_input_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_scaled')
    shifted_input_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_shifted')  # Main op's input

    container.add_initializer(alpha_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [alpha])
    container.add_initializer(beta_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [beta])

    # Compose an Affine preprocessing pipeline
    container.add_node('Mul', [operator.inputs[0].full_name, alpha_tensor_name], scaled_input_name,
                       name=scope.get_unique_operator_name('Mul'), broadcast=1)
    container.add_node('Add', [operator.inputs[0].full_name, beta_tensor_name], shifted_input_name,
                       name=scope.get_unique_operator_name('Add'), broadcast=1)

    simple_unary_map = {Params.SQRT: 'Sqrt', Params.INVERSE: 'Reciprocal',
                        Params.EXP: 'Exp', Params.LOG: 'Log', Params.ABS: 'Abs'}

    if params.type == Params.RSQRT:
        op_type = 'Sqrt'
        sqrt_op_name = scope.get_unique_operator_name(op_type)
        sqrt_name = scope.get_unique_variable_name(op_type + '_output')
        container.add_node(op_type, [shifted_input_name], [sqrt_name], name=sqrt_op_name)

        op_type = 'Reciprocal'
        inverse_op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [sqrt_name], operator.output_full_names, name=inverse_op_name)
    elif params.type == Params.POWER:
        exp_name = scope.get_unique_variable_name('Y')
        container.add_initializer(exp_name, onnx_proto.TensorProto.FLOAT, [1], [params.alpha])

        op_type = 'Pow'
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [shifted_input_name, exp_name], operator.output_full_names, name=op_name)
    elif params.type == Params.THRESHOLD:
        op_type = 'Clip'
        op_name = scope.get_unique_operator_name(op_type)
        attrs = {'name': op_name, 'min': params.alpha}
        container.add_node(op_type, shifted_input_name, operator.output_full_names, **attrs)
    elif params.type in simple_unary_map:
        op_type = simple_unary_map[params.type]
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, shifted_input_name, operator.output_full_names, name=op_name)
    else:
        raise ValueError('Unsupported unary function :{}'.format(params.type))


register_converter('unary', convert_unary)
