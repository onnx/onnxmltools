# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import *
from ....common._registration import register_converter


def convert_unary(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UnaryFunctionLayerParams as Params

    params = operator.raw_operator.unary
    preprocessor_name = scope.get_unique_operator_name('Affine')
    preprocessed_variable_name = scope.get_unique_variable_name(preprocessor_name + '_output')
    apply_affine(scope, operator.input_full_names[0], preprocessed_variable_name, container,
                 operator_name=preprocessor_name, alpha=params.scale, beta=params.shift)

    if params.type == Params.RSQRT:
        sqrt_tensor_name = scope.get_unique_variable_name(operator.full_name + '_intra_tensor')
        apply_sqrt(scope, preprocessed_variable_name, sqrt_tensor_name, container)
        apply_reciprocal(scope, sqrt_tensor_name, operator.output_full_names, container)
    elif params.type == Params.POWER:
        exp_name = scope.get_unique_variable_name('Y')
        container.add_initializer(exp_name, onnx_proto.TensorProto.FLOAT, [], [params.alpha])

        apply_pow(scope, [preprocessed_variable_name, exp_name], operator.output_full_names, container,
                  operator_name=operator.full_name, broadcast=1)
    elif params.type == Params.THRESHOLD:
        apply_clip(scope, preprocessed_variable_name, operator.output_full_names, container,
                   operator_name=operator.full_name, min=params.alpha)
    elif params.type == Params.SQRT:
        apply_sqrt(scope, preprocessed_variable_name, operator.output_full_names, container,
                   operator_name=operator.full_name)
    elif params.type == Params.INVERSE:
        apply_reciprocal(scope, preprocessed_variable_name, operator.output_full_names, container,
                         operator_name=operator.full_name)
    elif params.type == Params.EXP:
        apply_exp(scope, preprocessed_variable_name, operator.output_full_names, container,
                  operator_name=operator.full_name)
    elif params.type == Params.LOG:
        apply_log(scope, preprocessed_variable_name, operator.output_full_names, container,
                  operator_name=operator.full_name)
    elif params.type == Params.ABS:
        apply_abs(scope, preprocessed_variable_name, operator.output_full_names, container,
                  operator_name=operator.full_name)
    else:
        raise ValueError('Unsupported unary function :{}'.format(params.type))


register_converter('unary', convert_unary)
