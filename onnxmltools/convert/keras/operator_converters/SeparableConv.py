# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy
import keras
from keras.layers import SeparableConv2D
from distutils.version import StrictVersion
if StrictVersion(keras.__version__) >= StrictVersion('2.1.3'):
    from keras.layers import SeparableConv1D

from ....proto import onnx_proto
from ...common._apply_operation import apply_identity, apply_transpose
from ...common._registration import register_converter
from .Dense import _activation_map


def convert_keras_separable_conv_core(scope, operator, container, is_transpose, n_dims, input_perm_axes,
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

    op_type = 'Conv'
    convolution_input_names_0 = [adjusted_input_name]
    attrs_0 = {'name': operator.full_name + '0'}

    parameters = op.get_weights()
    assert (len(parameters) == 3 if op.use_bias else 2)
    weight_params = parameters[0]

    kernel_size = weight_params.shape[:-2]
    assert (kernel_size == op.kernel_size)

    attrs_0['group'] = weight_params.shape[-2]
    shape = weight_params.shape
    new_shape = shape[:-2] + (1, shape[-2] * shape[-1])
    weight_params = numpy.reshape(weight_params, new_shape).transpose(weight_perm_axes)

    weight_tensor_name = scope.get_unique_variable_name('W_0')
    container.add_initializer(weight_tensor_name, onnx_proto.TensorProto.FLOAT,
                              weight_params.shape, weight_params.flatten())
    convolution_input_names_0.append(weight_tensor_name)

    attrs_0['dilations'] = list(op.dilation_rate)
    attrs_0['strides'] = list(op.strides)
    attrs_0['kernel_shape'] = op.kernel_size

    if op.padding == 'valid':
        attrs_0['auto_pad'] = 'VALID'
    elif op.padding == 'same':
        if is_transpose:  # bypass onnx engine issue on convtranpose support.
            attrs_0['auto_pad'] = 'SAME_LOWER'
            shape = [-1 if i is None else i for i in op.output_shape]
            if channels_first:
                attrs_0['output_shape'] = shape
            else:
                attrs_0['output_shape'] = shape[0:1] + shape[-1:] + shape[1:-1]

        else:
            attrs_0['auto_pad'] = 'SAME_LOWER'
    else:
        raise RuntimeError("Unsupported padding type '{}'".format(op.padding))

    intermediate_output_name_0 = scope.get_unique_variable_name('convolution_output_0')
    container.add_node(op_type, convolution_input_names_0,
                       intermediate_output_name_0, **attrs_0)

    # the second Conv
    convolution_input_names_1 = [intermediate_output_name_0]
    attrs_1 = {'name': operator.full_name + '1'}

    weight_tensor_name = scope.get_unique_variable_name('W_1')
    weight_params = parameters[1].transpose(weight_perm_axes)
    container.add_initializer(weight_tensor_name, onnx_proto.TensorProto.FLOAT,
                              weight_params.shape, weight_params.flatten())
    convolution_input_names_1.append(weight_tensor_name)

    if len(parameters) == 3:
        bias_tensor_name = scope.get_unique_variable_name('B_1')
        container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT,
                                  parameters[2].shape, parameters[2].flatten())
        convolution_input_names_1.append(bias_tensor_name)

    all_ones = numpy.ones(n_dims, numpy.int8)
    attrs_1['dilations'] = all_ones
    attrs_1['strides'] = all_ones
    attrs_1['kernel_shape'] = all_ones
    attrs_1['group'] = 1
    attrs_1['auto_pad'] = attrs_0['auto_pad']

    intermediate_output_name_1 = scope.get_unique_variable_name('convolution_output_1')
    container.add_node(op_type, convolution_input_names_1,
                       intermediate_output_name_1, **attrs_1)

    # The construction of convolution is done. Now, we create an activation operator to apply the activation specified
    # in this Keras layer.
    apply_activation_function = _activation_map[op.activation]
    activation_output_name = scope.get_unique_variable_name('activation_output')
    apply_activation_function(scope, intermediate_output_name_1, activation_output_name, container)

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


def convert_keras_separable_conv1d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(1, False)
    convert_keras_separable_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)


def convert_keras_separable_conv2d(scope, operator, container):
    is_transpose, n_dims, input_perm, output_perm, weight_perm = get_converter_config(2, False)
    convert_keras_separable_conv_core(scope, operator, container, is_transpose, n_dims, input_perm, output_perm, weight_perm)

if StrictVersion(keras.__version__) >= StrictVersion('2.1.5'):
    register_converter(SeparableConv1D, convert_keras_separable_conv1d)
    register_converter(SeparableConv2D, convert_keras_separable_conv2d)
