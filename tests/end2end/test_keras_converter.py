# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnxmltools
from onnxmltools.utils import dump_data_and_model
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

    def call(self, inputs, **kwargs):
        return self.alpha * K.tanh(self.beta * inputs)

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
        converted_model = onnxmltools.convert_keras(model, custom_conversion_functions={ScaledTanh: custom_activation})

        actual = model.predict(x)
        self.assertIsNotNone(actual)

        self.assertIsNotNone(converted_model)
        # to check the model, you can print(str(converted_model))
        dump_data_and_model(x.astype(np.float32), model, converted_model, basename="KerasCustomOp-Out0",
                            context=dict(ScaledTanh=ScaledTanh))

    def test_channel_last(self):
        N, C, H, W = 2, 3, 5, 5
        x = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        model = keras.Sequential()
        model.add(Conv2D(2, kernel_size=(1, 2), strides=(1, 1), padding='valid', input_shape=(H, W, C),
                         data_format='channels_last'))  # , activation='softmax')
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last'))

        model.compile(optimizer='sgd', loss='mse')
        converted_model = onnxmltools.convert_keras(model, channel_first_inputs=[model.inputs[0].name])

        expected = model.predict(x)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(converted_model)

        try:
            import onnxruntime
            sess = onnxruntime.InferenceSession(converted_model.SerializeToString())
            actual = sess.run([], {sess.get_inputs()[0].name:
                                         np.transpose(x.astype(np.float32), [0, 3, 1, 2])})
            self.assertTrue(np.allclose(expected, actual))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
