# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import onnxmltools
import numpy as np
import unittest
import keras

from keras import backend as K
from keras.layers import *

class TestKerasConverter(unittest.TestCase):
    def test_model_creation(self):
        N, C, H, W = 2, 3, 5, 5
        input1 = keras.layers.Input(shape=(H, W, C))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(H, W, C))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        maximum_layer = keras.layers.Maximum()([x1, x2])

        out = keras.layers.Dense(8)(maximum_layer)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)

        trial1 = np.random.rand(N, H, W, C).astype(np.float32, copy=False)
        trial2 = np.random.rand(N, H, W, C).astype(np.float32, copy=False)

        predicted = model.predict([trial1, trial2])
        self.assertIsNotNone(predicted)

        converted_model_8 = onnxmltools.convert_keras(model, target_opset=8)
        converted_model_4 = onnxmltools.convert_keras(model, target_opset=4)

        self.assertIsNotNone(converted_model_8)
        self.assertIsNotNone(converted_model_4)

        opset_comparison = converted_model_8.opset_import[0].version > converted_model_4.opset_import[0].version

        self.assertTrue(opset_comparison)

if __name__ == "__main__":
    unittest.main()