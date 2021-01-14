# SPDX-License-Identifier: Apache-2.0

from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_add, apply_mul


def convert_preprocessing_scaler(scope, operator, container):
    params = operator.raw_operator.scaler
    # Specify some of this operator's attribute. The scale parameter in CoreML is always a scalar.
    # We just copy it and let ONNX scaler to broadcast it to all channels.

    color_space = operator.inputs[0].type.color_space
    if color_space == 'Gray8':
        bias = [params.grayBias]
    elif color_space == 'Rgb8':
        bias = [params.redBias, params.greenBias, params.blueBias]
    elif color_space == 'Bgr8':
        bias = [params.blueBias, params.greenBias, params.redBias]
    else:
        raise ValueError('Unknown color space for tensor {}'.format(operator.inputs[0].full_name))

    if container.target_opset < 9:
        attrs = {'name': operator.full_name, 'scale': params.channelScale}
        attrs['bias'] = bias
        container.add_node('ImageScaler', [operator.inputs[0].full_name], [operator.outputs[0].full_name], **attrs)
    else:
        # In comments below, assume input tensor is X, the scale scalar is a, the bias vector is b.

        # Store the scalar, a, used to scale all elements in the input tensor.
        aName = scope.get_unique_variable_name(operator.full_name + '_scale')
        container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, [1], [params.channelScale])

        # Store the bias vector. It will be added into the input tensor.
        bName = scope.get_unique_variable_name(operator.full_name + '_bias')
        container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, [len(bias), 1, 1], bias)

        # Compute Z = a * X.
        zName = scope.get_unique_variable_name(operator.full_name + '_scaled')
        apply_mul(scope, [operator.input_full_names[0], aName], zName, container)

        # Compute Y = Z + b, which is the final output.
        apply_add(scope, [bName, zName], operator.output_full_names[0], container)


register_converter('scalerPreprocessor', convert_preprocessing_scaler)
