import sys

import onnxmltools
import numpy as np
import unittest
import keras

from keras import backend as K
from keras.layers import *


class ScaledTanh(keras.layers.Layer):
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super(ScaledTanh, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def build(self, input_shape):
        super(ScaledTanh, self).build(input_shape)

    def call(self, x):
        return self.alpha * K.tanh(self.beta * x)

    def compute_output_shape(self, input_shape):
        return input_shape


def custom_activation(scope, operator, container):
    # type:(ScopeBase, OperatorBase, ModelContainer) -> None
    container.add_node('ScaledTanh', operator.input_full_names, operator.output_full_names,
                       op_version=1, alpha=operator.original_operator.alpha, beta=operator.original_operator.beta)


class TestKerasConverter(unittest.TestCase):
    def test_custom_op(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = keras.Sequential()
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))
        model.add(ScaledTanh(0.9, 2.0))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

        model.compile(optimizer='sgd', loss='mse')
        actual = model.predict(x)
        self.assertIsNotNone(actual)

        converted_model = onnxmltools.convert_keras(model, custom_conversion_functions={ScaledTanh: custom_activation})
        self.assertIsNotNone(converted_model)
        # to check the model, you can print(str(converted_model))
