# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from distutils.version import StrictVersion
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D
from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D
from ...common._apply_operation import apply_reshape
from ...common._apply_operation import apply_transpose
from ...common._registration import register_converter
from .common import get_permutation_config


def convert_keras_pooling_core(scope, operator, container, is_global, n_dims,
                               op_type, input_perm_axes, output_perm_axes):
    op = operator.raw_operator
    channels_first = n_dims > 1 and op.data_format == 'channels_first'

    # TODO: extract this piece of code to be a common method.
    if channels_first:
        adjusted_pooling_input = operator.inputs[0].full_name
    else:
        adjusted_pooling_input = scope.get_unique_variable_name('input_transposed')
        apply_transpose(scope, operator.inputs[0].full_name, adjusted_pooling_input, container, perm=input_perm_axes)

    op_type_prefix = 'Global' if is_global else ''
    if op_type == 'Avg':
        onnx_op_type = 'AveragePool'
        if operator.targeted_onnx_version < StrictVersion('1.2'):
            op_version = 1
        else:
            op_version = 7
    elif op_type == 'Max':
        onnx_op_type = 'MaxPool'
        op_version = 1
    else:
        raise RuntimeError('Unsupported Keras pooling type: %s' % op_type)

    attrs = {'name': operator.full_name}
    if not is_global:
        attrs['strides'] = list(op.strides)
        attrs['kernel_shape'] = op.pool_size
        if op.padding == 'valid':
            attrs['auto_pad'] = 'VALID'
        elif op.padding == 'same':
            attrs['auto_pad'] = 'SAME_LOWER'
        else:
            raise RuntimeError("Unsupported padding type '{0}'".format(op.padding))

    # The output_tensor_name is used to store the Keras result produced by ONNX operators.
    # For global pooling, a Reshape op is needed to match the actual Keras's output shape.
    output_tensor_name = scope.get_unique_variable_name('pooling_for_reshape') if is_global else operator.outputs[0].full_name

    if channels_first:
        # In this case, the output of our Pool operator just match what Keras produces.
        container.add_node(op_type_prefix + onnx_op_type, adjusted_pooling_input,
                           output_tensor_name, op_version=op_version, **attrs)
    else:
        # Put the output of Pool operator to an intermediate tensor. Laster we will apply a Transpose to match the
        # original Keras output format
        pooling_output_name = scope.get_unique_variable_name('pooling_output')
        container.add_node(op_type_prefix + onnx_op_type, adjusted_pooling_input, pooling_output_name,
                           op_version=op_version, **attrs)

        # Generate a final Transpose
        apply_transpose(scope, pooling_output_name, output_tensor_name, container, perm=output_perm_axes)

    if is_global:
        apply_reshape(scope, output_tensor_name,
                      operator.outputs[0].full_name, container, desired_shape=operator.outputs[0].type.shape)


def convert_keras_max_pooling_1d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(1)
    convert_keras_pooling_core(scope, operator, container, is_global=False, n_dims=1, op_type='Max',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_max_pooling_2d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(2)
    convert_keras_pooling_core(scope, operator, container, is_global=False, n_dims=2, op_type='Max',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_max_pooling_3d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(2)
    convert_keras_pooling_core(scope, operator, container, is_global=False, n_dims=3, op_type='Max',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_average_pooling_1d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(1)
    convert_keras_pooling_core(scope, operator, container, is_global=True, n_dims=1, op_type='Avg',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_average_pooling_2d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(2)
    convert_keras_pooling_core(scope, operator, container, is_global=False, n_dims=2, op_type='Avg',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_average_pooling_3d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(3)
    convert_keras_pooling_core(scope, operator, container, is_global=False, n_dims=3, op_type='Avg',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_global_max_pooling_1d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(1)
    convert_keras_pooling_core(scope, operator, container, is_global=True, n_dims=1, op_type='Max',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_global_max_pooling_2d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(2)
    convert_keras_pooling_core(scope, operator, container, is_global=True, n_dims=2, op_type='Max',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_global_average_1d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(1)
    convert_keras_pooling_core(scope, operator, container, is_global=True, n_dims=1, op_type='Avg',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


def convert_keras_global_average_2d(scope, operator, container):
    input_perm_axes, output_perm_axes = get_permutation_config(2)
    convert_keras_pooling_core(scope, operator, container, is_global=True, n_dims=2, op_type='Avg',
                               input_perm_axes=input_perm_axes, output_perm_axes=output_perm_axes)


register_converter(MaxPooling1D, convert_keras_max_pooling_1d)
register_converter(MaxPooling2D, convert_keras_max_pooling_2d)
register_converter(MaxPooling3D, convert_keras_max_pooling_3d)

register_converter(AveragePooling1D, convert_keras_average_pooling_1d)
register_converter(AveragePooling2D, convert_keras_average_pooling_2d)
register_converter(AveragePooling3D, convert_keras_average_pooling_3d)

register_converter(GlobalMaxPooling1D, convert_keras_global_max_pooling_1d)
register_converter(GlobalMaxPooling2D, convert_keras_global_max_pooling_2d)

register_converter(GlobalAveragePooling1D, convert_keras_global_average_1d)
register_converter(GlobalAveragePooling2D, convert_keras_global_average_2d)
