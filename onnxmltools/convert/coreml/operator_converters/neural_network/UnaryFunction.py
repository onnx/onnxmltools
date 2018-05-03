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
    preprocessor_type = 'Affine'
    preprocessor_name = scope.get_unique_operator_name(preprocessor_type)
    preprocessor_attrs = {'name': preprocessor_name, 'alpha': params.scale, 'beta': params.shift}

    preprocessed_variable_name = scope.get_unique_variable_name(preprocessor_name + '_output')
    container.add_node(preprocessor_type, operator.input_full_names, [preprocessed_variable_name], **preprocessor_attrs)

    simple_unary_map = {Params.SQRT: 'Sqrt', Params.INVERSE: 'Reciprocal',
                        Params.EXP: 'Exp', Params.LOG: 'Log', Params.ABS: 'Abs'}

    if params.type == Params.RSQRT:
        op_type = 'Sqrt'
        sqrt_op_name = scope.get_unique_operator_name(op_type)
        sqrt_name = scope.get_unique_variable_name(op_type + '_output')
        container.add_node(op_type, [preprocessed_variable_name], [sqrt_name], name=sqrt_op_name)

        op_type = 'Reciprocal'
        inverse_op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [sqrt_name], operator.output_full_names, name=inverse_op_name)
    elif params.type == Params.POWER:
        exp_name = scope.get_unique_variable_name('Y')
        container.add_initializer(exp_name, onnx_proto.TensorProto.FLOAT, [1], [params.alpha])

        op_type = 'Pow'
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, [operator.inputs[0].full_name, exp_name], operator.output_full_names, name=op_name)
    elif params.type == Params.THRESHOLD:
        op_type = 'Clip'
        op_name = scope.get_unique_operator_name(op_type)
        attrs = {'name': op_name, 'min': params.alpha}
        container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)
    elif params.type in simple_unary_map:
        op_type = simple_unary_map[params.type]
        op_name = scope.get_unique_operator_name(op_type)
        container.add_node(op_type, operator.input_full_names, operator.output_full_names, name=op_name)
    else:
        raise ValueError('Unsupported unary function :{}'.format(params.type))


register_converter('unary', convert_unary)
