# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, RepeatVector
from ...common._registration import register_shape_calculator


def calculate_keras_conv_output_shapes(operator):
    if operator.type not in [Conv2DTranspose, Conv3DTranspose]:
        return
    op = operator.raw_operator
    operator.outputs[0].type.shape = list(i if isinstance(i, numbers.Integral) else 'None' for i in op.output_shape)


# Convolution
register_shape_calculator(Conv1D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv2D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv3D, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv2DTranspose, calculate_keras_conv_output_shapes)
register_shape_calculator(Conv3DTranspose, calculate_keras_conv_output_shapes)
