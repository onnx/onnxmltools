# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import keras
from ...common._registration import register_shape_calculator

def calculate_keras_global_pooling_output_shapes(operator):
    d_len = len(operator.raw_operator.input_shape)
    d_app = [1 for i in range(d_len -2)]
    operator.outputs[0].type.shape = operator.outputs[0].type.shape + d_app


# Global Pooling
register_shape_calculator(keras.layers.GlobalMaxPooling1D, calculate_keras_global_pooling_output_shapes)
register_shape_calculator(keras.layers.GlobalMaxPooling2D, calculate_keras_global_pooling_output_shapes)
register_shape_calculator(keras.layers.GlobalMaxPooling3D, calculate_keras_global_pooling_output_shapes)

register_shape_calculator(keras.layers.GlobalAveragePooling1D, calculate_keras_global_pooling_output_shapes)
register_shape_calculator(keras.layers.GlobalAveragePooling2D, calculate_keras_global_pooling_output_shapes)
register_shape_calculator(keras.layers.GlobalAveragePooling3D, calculate_keras_global_pooling_output_shapes)
