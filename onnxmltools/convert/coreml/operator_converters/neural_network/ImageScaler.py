# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_preprocessing_scaler(scope, operator, container):
    params = operator.raw_operator.scaler
    # Determine some of this operator's attribute. The scale parameter in CoreML is always a scalar.
    # We just copy it and let ONNX scaler to broadcast it to all channels. For bias, its format depends on the input
    # image's format.
    scale = [params.channelScale]
    color_space = operator.inputs[0].type.color_space
    if color_space == 'GRAY':
        bias = [params.grayBias]
    elif color_space == 'RGB':
        bias = [params.redBias, params.greenBias, params.blueBias]
    elif color_space == 'BGR':
        bias = [params.blueBias, params.greenBias, params.redBias]
    else:
        raise ValueError('Unknown color space for tensor {}'.format(operator.inputs[0].full_name))

    scale_tensor_name = scope.get_unique_operator_name(operator.full_name + '_scale')
    bias_tensor_name = scope.get_unique_operator_name(operator.full_name + '_bias')
    intra_tensor_name = scope.get_unique_operator_name(operator.inputs[0].full_name + '_scaled')

    container.add_initializer(scale_tensor_name, onnx_proto.TensorProto.FLOAT, [], scale)
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, [len(bias)], bias)

    container.add_node('Mul', [operator.inputs[0].full_name, scale_tensor_name], intra_tensor_name,
                       name=scope.get_unique_operator_name('Mul'), broadcast=1)

    if len(bias) > 1:
        container.add_node('Add', [intra_tensor_name, bias_tensor_name], operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Add'), broadcast=1, axis=1)
    else:
        container.add_node('Add', [intra_tensor_name, bias_tensor_name], operator.outputs[0].full_name,
                           name=scope.get_unique_operator_name('Add'), broadcast=1)


register_converter('scalerPreprocessor', convert_preprocessing_scaler)
