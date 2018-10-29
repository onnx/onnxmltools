"""
Tests scikit-onehotencoder converter.
"""
import unittest
from sklearn.preprocessing import OneHotEncoder
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType

class TestSklearnOneHotEncoderConverter(unittest.TestCase):

    def test_model_one_hot_encoder(self):
        # categorical_features will be removed in 0.22 (this test will fail by then).
        model = OneHotEncoder()
        model.fit([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]])
        model_onnx = convert_sklearn(model, 'scikit-learn one-hot encoder', [('input', Int64TensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_one_hot_encoder_mixed_float_int(self):
        # categorical_features will be removed in 0.22 (this test will fail by then).
        model = OneHotEncoder()
        model.fit([[0.4, 0.2, 3], [1.4, 1.2, 0], [0.2, 2.2, 1]])
        model_onnx = convert_sklearn(model, 'one-hot encoder mixed-type inputs',
                               [('input1', FloatTensorType([1, 2])), ('input2', Int64TensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)


if __name__ == "__main__":
    unittest.main()
