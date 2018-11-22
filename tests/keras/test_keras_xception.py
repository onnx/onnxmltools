import sys
from distutils.version import StrictVersion
import onnxmltools
from onnxmltools.utils import dump_data_and_model
import numpy as np
import unittest
try:
    import keras
    has_keras = True
except ImportError:
    has_keras = False
    
if has_keras:
    from keras.applications.xception import Xception as NNKeras
    from keras import backend as K
    from keras.layers import *


class TestKerasConverterXCeption(unittest.TestCase):
    
    @unittest.skip(reason="Unsupported shape calculation for operator SeparableConv2D'>")
    def test_xception(self):        
        x = np.random.rand(1, 224, 224, 3).astype(np.float32, copy=False)
        model = NNKeras(input_shape=None,
                        include_top=True, weights='imagenet',
                        input_tensor=None, pooling=None, classes=1000)
        converted_model = onnxmltools.convert_keras(model)
        self.assertIsNotNone(converted_model)
        dump_data_and_model(x, model, converted_model, basename="KerasMobileNet")


if __name__ == "__main__":
    unittest.main()
