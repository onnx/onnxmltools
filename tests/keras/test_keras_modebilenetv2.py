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
    try:
        from keras.applications.mobilenetv2 import MobileNetV2 as MobileNet
    except ModuleNotFoundError:
        from keras.applications.mobilenet import MobileNet

    from keras import backend as K
    from keras.layers import *


class TestKerasConverterMobileNetv2(unittest.TestCase):
    
    @unittest.skipIf(not has_keras or StrictVersion(keras.__version__) >= StrictVersion('2.2'),
                     reason="Unsupported shape calculation for operator <class 'keras.layers.advanced_activations.ReLU'>")
    def test_mobilenetv2(self):
        x = np.random.rand(1, 224, 224, 3).astype(np.float32, copy=False)
        model = MobileNet(input_shape=None, alpha=1.0,
                          depth_multiplier=1,
                          include_top=True, weights='imagenet',
                          input_tensor=None, pooling=None, classes=1000)
        converted_model = onnxmltools.convert_keras(model)
        self.assertIsNotNone(converted_model)

        # runtime fails
        # onnxruntime 3.3:
        # Method run failed due to: [LotusError] : 1 : GENERAL ERROR : 
        # onnxruntime/core/providers/cpu/tensor/pad.cc:53 onnxruntime::common::Status onnxruntime::Pad<T>::Compute(onnxruntime::OpKernelContext*) const [with T = float] dimension_count * 2 == pads_.size() was false.
        # 'pads' attribute has wrong number of values
        # onnxruntime 3.3+
        # 1 : GENERAL ERROR : onnxruntime/core/framework/op_kernel.cc:40 onnxruntime::OpKernelContext::Output status.IsOK() was false. Tensor shape cannot contain any negative

        # dump_data_and_model(x, model, converted_model, basename="KerasMobileNet")


if __name__ == "__main__":
    unittest.main()
