# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras.layers
from ...common._apply_operation import apply_transpose, apply_upsample
from ...common._registration import register_converter
from .common import get_permutation_config


def convert_keras_upsample(scope, operator, container, n_dims):
    op = operator.raw_operator
    height_scale = int(op.size[0])
    width_scale = int(op.size[1])
    scales = [1, 1, height_scale, width_scale]

    # Derive permutation configuration. If the Keras input format is not channels_first, this configuration may be used
    # to manipulate the input and output of ONNX Upsample.
    input_perm_axes, output_perm_axes = get_permutation_config(n_dims)
    channels_first = n_dims > 1 and op.data_format == 'channels_first'

    # Before creating the main Upsample operator, we need to permute the input tensor if the original operator is
    # working under channels_last mode.
    if channels_first:
        # No permutation is required. Use input as it is.
        input_tensor_name = operator.inputs[0].full_name
    else:
        # Permute the original input and then use the permuted result as the input of ONNX Upsample
        input_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_permuted')
        apply_transpose(scope, operator.inputs[0].full_name, input_tensor_name, container, perm=input_perm_axes)

    # If channels_first is True, we don't need to permute the output of ONNX Upsample. Otherwise, similar to Crop's
    # conversion, a Transpose would be added.
    if channels_first:
        apply_upsample(scope, input_tensor_name, operator.outputs[0].full_name, container, scales=scales)
    else:
        upsampled_tensor_name = scope.get_unique_variable_name(input_tensor_name + '_upsampled')
        apply_upsample(scope, input_tensor_name, upsampled_tensor_name, container, scales=scales)
        apply_transpose(scope, upsampled_tensor_name, operator.outputs[0].full_name, container, perm=output_perm_axes)


def convert_keras_upsample_1d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=1)


def convert_keras_upsample_2d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=2)


def convert_keras_upsample_3d(scope, operator, container):
    convert_keras_upsample(scope, operator, container, n_dims=3)


register_converter(keras.layers.UpSampling1D, convert_keras_upsample_1d)
register_converter(keras.layers.UpSampling2D, convert_keras_upsample_2d)
register_converter(keras.layers.UpSampling3D, convert_keras_upsample_3d)
