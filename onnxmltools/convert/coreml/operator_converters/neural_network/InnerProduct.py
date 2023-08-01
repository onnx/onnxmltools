# SPDX-License-Identifier: Apache-2.0

from .....proto import onnx_proto
from ....common._apply_operation import apply_reshape
from ....common._registration import register_converter


def convert_inner_product(scope, operator, container):
    params = operator.raw_operator.innerProduct

    # Apply pre-processing step if needed
    if len(operator.inputs[0].type.shape) == 4:
        # Input shape is [N, C, 1, 1]. Adjust input shape because Gemm in ONNX only takes 2-D input
        reshaped_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_reshaped')
        apply_reshape(scope, operator.inputs[0].full_name, reshaped_tensor_name, container,
                      desired_shape=[-1, int(params.inputChannels)])
        name_a = reshaped_tensor_name
    else:
        # Input shape is [N, C]. There is no pre-processing for applying ONNX operator.
        name_a = operator.inputs[0].full_name

    # Allocate the weights of dense layer
    name_b = scope.get_unique_variable_name(operator.full_name + '_B')
    shape_b = [params.outputChannels, params.inputChannels]
    container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.weights.floatValue)

    # Allocate the bias of dense layer
    name_c = scope.get_unique_variable_name(operator.full_name + '_C')
    shape_c = [params.outputChannels]
    if params.hasBias:
        container.add_initializer(name_c, onnx_proto.TensorProto.FLOAT, shape_c, params.bias.floatValue)
    else:
        container.add_initializer(name_c, onnx_proto.TensorProto.FLOAT, shape_c, [0.] * shape_b[0])

    # Set up attributes for ONNX Gemm which is the counterpart of CoreML inner product layer in ONNX.
    attrs = {'name': operator.full_name}
    attrs['alpha'] = 1.0
    attrs['beta'] = 1.0
    attrs['transA'] = 0
    attrs['transB'] = 1

    # Get the correct version number for Gemm in ONNX
    if container.target_opset < 5:
        attrs['broadcast'] = 1
        op_version = 1
    elif container.target_opset < 7:
        attrs['broadcast'] = 1
        op_version = 6
    elif container.target_opset < 9:
        op_version = 7
    elif container.target_opset < 11:
        op_version = 9
    else:
        op_version = 11

    # Create the major ONNX operator, Gemm, to do CoreML inner product and possibly add shape adjustment
    if len(operator.inputs[0].type.shape) == 4:
        # Input shape is [N, C, 1, 1] so we expect output is also 4-D, [N, C', 1, 1].
        buffer_tensor_name = scope.get_unique_variable_name(operator.full_name + '_buffer')
        container.add_node('Gemm', [name_a, name_b, name_c], buffer_tensor_name, op_version=op_version, **attrs)
        apply_reshape(scope, buffer_tensor_name, operator.outputs[0].full_name, container,
                      desired_shape=[-1, int(params.outputChannels), 1, 1])
    else:
        # Input shape is [N, C], so we don't need to change Gemm's output shape.
        container.add_node('Gemm', [name_a, name_b, name_c], operator.outputs[0].full_name,
                           op_version=op_version, **attrs)

register_converter('innerProduct', convert_inner_product)
