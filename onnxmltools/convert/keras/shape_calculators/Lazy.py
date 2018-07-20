# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from ...common._registration import register_shape_calculator


def calculate_lazy_output_shapes(operator):
    pass


# Activation
register_shape_calculator(keras.layers.Activation, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.advanced_activations.LeakyReLU, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.advanced_activations.ThresholdedReLU, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.advanced_activations.ELU, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.advanced_activations.PReLU, calculate_lazy_output_shapes)
# TODO:Following layer is not supported by the checked-in keras version and requires an upgrade of the checked-in keras
# register_shape_calculator(keras.layers.advanced_activations.Softmax, calculate_lazy_output_shapes)

# Concate
register_shape_calculator(keras.layers.Concatenate, calculate_lazy_output_shapes)

# Cropping
register_shape_calculator(keras.layers.Cropping1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Cropping2D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Cropping3D, calculate_lazy_output_shapes)

# Local Pooling
register_shape_calculator(keras.layers.MaxPooling1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.MaxPooling2D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.MaxPooling3D, calculate_lazy_output_shapes)

register_shape_calculator(keras.layers.AveragePooling1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.AveragePooling2D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.AveragePooling3D, calculate_lazy_output_shapes)

# Global Pooling
register_shape_calculator(keras.layers.GlobalMaxPooling1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.GlobalMaxPooling2D, calculate_lazy_output_shapes)

register_shape_calculator(keras.layers.GlobalAveragePooling1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.GlobalAveragePooling2D, calculate_lazy_output_shapes)

# Flatten
register_shape_calculator(keras.layers.Flatten, calculate_lazy_output_shapes)

# Dense
register_shape_calculator(keras.layers.Dense, calculate_lazy_output_shapes)

# Dot
register_shape_calculator(keras.layers.Dot, calculate_lazy_output_shapes)

# BatchNorm
register_shape_calculator(keras.layers.BatchNormalization, calculate_lazy_output_shapes)

# Merge
register_shape_calculator(keras.layers.Add, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Multiply, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Subtract, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Average, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.Maximum, calculate_lazy_output_shapes)

# Permute
register_shape_calculator(keras.layers.Permute, calculate_lazy_output_shapes)

# Reshape
register_shape_calculator(keras.layers.core.Reshape, calculate_lazy_output_shapes)

# Upsample
register_shape_calculator(keras.layers.UpSampling1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.UpSampling2D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.UpSampling3D, calculate_lazy_output_shapes)

# ZeroPad
register_shape_calculator(keras.layers.ZeroPadding1D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.ZeroPadding2D, calculate_lazy_output_shapes)
register_shape_calculator(keras.layers.ZeroPadding3D, calculate_lazy_output_shapes)

# RepeatVector
register_shape_calculator(keras.layers.RepeatVector, calculate_lazy_output_shapes)
