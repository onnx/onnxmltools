# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from distutils.version import StrictVersion
import numbers
from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, RepeatVector
if StrictVersion(keras.__version__) >= StrictVersion('2.1.5'):
    from keras.layers import DepthwiseConv2D
from ...common._registration import register_shape_calculator


def calculate_keras_conv_output_shapes(operator):
    if operator.type not in [Conv2DTranspose, Conv3DTranspose]:
        return
    op = operator.raw_operator
    operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None' for i in op.output_shape)


def calculate_keras_depthwise_conv_output_shapes(operator):
    op = operator.raw_operator
    depth_multiplier = op.depth_multiplier
    if len(op.output_shape) != 4:
        raise ValueError("Number of dimensions of shape {0} must be 4.".format(op.output_shape))
    if op.output_shape[0] is not None:
        raise ValueError("Unexpected shape when computing the output shape {0}".format(op.output_shape))
    operator.outputs[0].type.shape = [1] + list(op.output_shape[1:3]) + [op.output_shape[3] * depth_multiplier]


# Convolution
register_shape_calculator(Conv1D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv2D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv3D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv2DTranspose, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv3DTranspose, calculate_keras_conv_output_shapes)
if StrictVersion(keras.__version__) >= StrictVersion('2.1.5'):
    register_shape_calculator(DepthwiseConv2D, calculate_keras_depthwise_conv_output_shapes)
