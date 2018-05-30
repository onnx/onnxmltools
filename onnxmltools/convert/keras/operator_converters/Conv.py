# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from ....proto import onnx_proto
from ...common._apply_operation import apply_identity, apply_transpose
from ...common._registration import register_converter
from .Dense import _activation_map


def convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm_axes,
                            output_perm_axes, weight_perm_axes):
    op = operator.raw_operator

    channels_first = n_dims > 1 and op.data_format == 'channels_first'

    # Unless channels_first is the Keras data format, the inputs and weights in Keras v.s. ONNX
    # are reversed. This is annoying, and inefficient as we'll have to use transposes.
    if channels_first:
        adjusted_input_name = operator.inputs[0].full_name
    else:
        adjusted_input_name = scope.get_unique_variable_name('adjusted_input')
        apply_transpose(scope, operator.inputs[0].full_name, adjusted_input_name, container, perm=input_perm_axes)

    op_type = 'ConvTranspose' if is_transpose else 'Conv'
    convolution_input_names = [adjusted_input_name]
    attrs = {'name': operator.full_name}

    parameters = op.get_weights()
    assert (len(parameters) == 2 if op.use_bias else 1)
    weight_params = parameters[0]

    input_channels, output_channels = weight_params.shape[-2:]
    kernel_size = weight_params.shape[:-2]
    assert (kernel_size == op.kernel_size)
    weight_params = weight_params.transpose(weight_perm_axes)

    weight_tensor_name = scope.get_unique_variable_name('W')
    container.add_initializer(weight_tensor_name, onnx_proto.TensorProto.FLOAT,
                              weight_params.shape, weight_params.flatten())
    convolution_input_names.append(weight_tensor_name)

    if len(parameters) == 2:
        bias_tensor_name = scope.get_unique_variable_name('B')
        container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT,
                                  parameters[1].shape, parameters[1].flatten())
        convolution_input_names.append(bias_tensor_name)

    attrs['dilations'] = list(op.dilation_rate)
    attrs['strides'] = list(op.strides)
    attrs['kernel_shape'] = op.kernel_size
    # Fix this...
    attrs['group'] = 1

    if op.padding == 'valid':
        attrs['auto_pad'] = 'VALID'
    elif op.padding == 'same':
        attrs['auto_pad'] = 'SAME_LOWER'
    else:
        raise RuntimeError("Unsupported padding type '{}'".format(op.padding))

    intermediate_output_name = scope.get_unique_variable_name('convolution_output')
    container.add_node(op_type, convolution_input_names,
                       intermediate_output_name, **attrs)

    # The construction of convolution is done. Now, we create an activation operator to apply the activation specified
    # in this Keras layer.
    apply_activation_function = _activation_map[op.activation]
    activation_output_name = scope.get_unique_variable_name('activation_output')
    apply_activation_function(scope, intermediate_output_name, activation_output_name, container)

    # Permute the output back of its original format
    if not channels_first:
        # Generate a final transposer.
        apply_transpose(scope, activation_output_name, operator.outputs[0].full_name, container, perm=output_perm_axes)
    else:
        apply_identity(scope, activation_output_name, operator.outputs[0].full_name, container)


def get_converter_config(dims, is_conv_transpose):
    assert (dims in [1, 2, 3])
    input_perm = [0, dims + 1] + list(range(1, dims + 1))
    output_perm = [0] + list(range(2, dims + 2)) + [1]
    weight_perm = [dims + 1, dims] + list(range(dims))
    return is_conv_transpose, dims, input_perm, output_perm, weight_perm


def convert_keras_conv1d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(1, False)
    convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


def convert_keras_conv2d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(2, False)
    convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


def convert_keras_conv3d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(3, False)
    convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


def convert_keras_conv_transpose_2d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(2, True)
    convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


def convert_keras_conv_transpose_3d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(3, True)
    convert_keras_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


register_converter(Conv1D, convert_keras_conv1d)
register_converter(Conv2D, convert_keras_conv2d)
register_converter(Conv3D, convert_keras_conv3d)
register_converter(Conv2DTranspose, convert_keras_conv_transpose_2d)
register_converter(Conv3DTranspose, convert_keras_conv_transpose_3d)
