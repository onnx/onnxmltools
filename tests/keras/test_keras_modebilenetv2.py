import sys

import onnxmltools
from onnxmltools.utils import dump_data_and_model
import numpy as np
import unittest
import keras
try:
    from keras.applications.mobilenetv2 import MobileNetV2 as MobileNet
except ModuleNotFoundError:
    from keras.applications.mobilenet import MobileNet

from keras import backend as K
from keras.layers import *


class TestKerasConverterMobileNetv2(unittest.TestCase):
    def test_mobilenetv2(self):
        x = np.random.rand(1, 224, 224, 3).astype(np.float32, copy=False)
        model = MobileNet(input_shape=None, alpha=1.0,
                          depth_multiplier=1,
                          include_top=True, weights='imagenet',
                          input_tensor=None, pooling=None, classes=1000)
        converted_model = onnxmltools.convert_keras(model)
        self.assertIsNotNone(converted_model)
        dump_data_and_model(x, model, converted_model,
                            basename="KerasMobileNet")


if __name__ == "__main__":
    unittest.main()
