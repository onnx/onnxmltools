import sys

import onnxmltools
from onnxmltools.utils import dump_data_and_model
import numpy as np
import unittest
import keras
from keras.applications.mobilenetv2 import MobileNetV2

from keras import backend as K
from keras.layers import *


class TestKerasConverterMobileNetv2(unittest.TestCase):
    def test_mobilenetv2(self):

        model = MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1,
                            include_top=True,
                            weights='imagenet', input_tensor=None,
                            pooling=None, classes=1000)
        converted_model = onnxmltools.convert_keras(model)
        self.assertIsNotNone(converted_model)
        # to check the model, you can print(str(converted_model))
        # dump_data_and_model(x.astype(np.float32), model, converted_model, basename="KerasMobileNetv2")


if __name__ == "__main__":
    unittest.main()
