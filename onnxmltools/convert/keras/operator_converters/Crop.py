# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from ...common._apply_operation import apply_transpose
from ...common._registration import register_converter
from .common import get_permutation_config


def convert_keras_crop(scope, operator, container, n_dims):
    op = operator.raw_operator
    op_type = 'Crop'
    attrs = {'name': operator.full_name}

    input_perm_axes, output_perm_axes = get_permutation_config(n_dims)
    channels_first = n_dims > 1 and op.data_format == 'channels_first'

    # Before creating the main Crop operator, we need to permute the input tensor if the original operator is working
    # under channels_last mode.
    if channels_first:
        input_tensor_name = operator.inputs[0].full_name
    else:
        input_tensor_name = scope.get_unique_variable_name(operator.inputs[0].full_name + '_permuted')
        apply_transpose(scope, operator.inputs[0].full_name, input_tensor_name, container, perm=input_perm_axes)

    param = op.cropping
    if isinstance(param, int):
        param = [param, param]

    if len(param) == 2:
        if isinstance(param[0], int):
            attrs['scale'] = param
        elif len(param[0]) == 2 and len(param[1]) == 2:
            # If tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
            top = param[0][0]
            bottom = param[0][1]
            left = param[1][0]
            right = param[1][1]
            attrs['border'] = [left, top, right, bottom]
        else:
            raise RuntimeError('Unknown crop parameter %s in CroppingLayer' % str(param))
    else:
        raise RuntimeError('Unknown crop parameter %s in CroppingLayer' % str(param))

    if not channels_first:
        cropped_tensor_name = scope.get_unique_variable_name(input_tensor_name + '_cropped')
        container.add_node(op_type, input_tensor_name, cropped_tensor_name, **attrs)
        apply_transpose(scope, cropped_tensor_name, operator.outputs[0].full_name, container, perm=output_perm_axes)
    else:
        container.add_node(op_type, input_tensor_name, operator.outputs[0].full_name, **attrs)


def convert_keras_crop_1d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=1)


def convert_keras_crop_2d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=2)


def convert_keras_crop_3d(scope, operator, container):
    convert_keras_crop(scope, operator, container, n_dims=3)


register_converter(keras.layers.Cropping1D, convert_keras_crop_1d)
register_converter(keras.layers.Cropping2D, convert_keras_crop_2d)
register_converter(keras.layers.Cropping3D, convert_keras_crop_3d)
