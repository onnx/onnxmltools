# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras.layers
from ...common._apply_operation import apply_pad, apply_transpose
from ...common._registration import register_converter
from .common import get_permutation_config


def get_padding_config(op, n_dims):
    pads = op.padding
    if isinstance(pads, int):
        pads = [pads] * n_dims
    else:
        pads = list(pads)
    pads[:0] = [0, 0]

    extended_n_dims = n_dims + 2  # adding N, C
    if len(pads) == extended_n_dims:
        full_pads = [None] * extended_n_dims
        for i in range(extended_n_dims):
            if isinstance(pads[i], int):
                full_pads[i] = [pads[i]] * 2
            else:
                full_pads[i] = pads[i]
    else:
        raise RuntimeError("padding parameter's dim({0}) is not correct.".format(n_dims))

    # keras schema ((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))
    # ONNX's [x1_begin, x2_begin...x1_end, x2_end,...], need re-shuffle.
    onnx_pads = [None] * 2 * extended_n_dims
    for i in range(extended_n_dims):
        onnx_pads[i] = full_pads[i][0]
        onnx_pads[extended_n_dims + i] = full_pads[i][1]

    return onnx_pads


def convert_keras_zero_pad(scope, operator, container, n_dims):
    op = operator.raw_operator

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

    # Prepare attributes for ONNX Pad
    mode = 'constant'
    pads = get_padding_config(op, n_dims)

    # If channels_first is True, we don't need to permute the output of ONNX Upsample. Otherwise, similar to Crop's
    # conversion, a Transpose would be added.
    if channels_first:
        apply_pad(scope, input_tensor_name, operator.outputs[0].full_name, container, mode=mode, pads=pads, value=0.)
    else:
        intermediate_tensor_name = scope.get_unique_variable_name(input_tensor_name + '_padded')
        apply_pad(scope, input_tensor_name, intermediate_tensor_name, container, mode=mode, pads=pads, value=0.)
        apply_transpose(scope, intermediate_tensor_name, operator.outputs[0].full_name, container,
                        perm=output_perm_axes)


def convert_keras_zero_pad_1d(scope, operator, container):
    convert_keras_zero_pad(scope, operator, container, 1)


def convert_keras_zero_pad_2d(scope, operator, container):
    convert_keras_zero_pad(scope, operator, container, 2)


def convert_keras_zero_pad_3d(scope, operator, container):
    convert_keras_zero_pad(scope, operator, container, 3)


register_converter(keras.layers.ZeroPadding1D, convert_keras_zero_pad_1d)
register_converter(keras.layers.ZeroPadding2D, convert_keras_zero_pad_2d)
register_converter(keras.layers.ZeroPadding3D, convert_keras_zero_pad_3d)
