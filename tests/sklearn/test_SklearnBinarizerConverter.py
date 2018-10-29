"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
from sklearn.preprocessing import Binarizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnBinarizer(unittest.TestCase):

    def test_model_binarizer(self):
        model = Binarizer(threshold=0.5)
        model_onnx = convert_sklearn(model, 'scikit-learn binarizer', [('input', FloatTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(numpy.array([[1, 1]], dtype=numpy.float32),
                            model, model_onnx, basename="SklearnBinarizer-SkipDim1")


if __name__ == "__main__":
    unittest.main()
