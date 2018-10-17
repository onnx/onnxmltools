"""
Tests scikit-normalizer converter.
"""
import unittest
import numpy
from sklearn.preprocessing import Normalizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnNormalizerConverter(unittest.TestCase):

    def test_model_normalizer(self):
        model = Normalizer(norm='l2')
        model_onnx = convert_sklearn(model, 'scikit-learn normalizer', [('input', Int64TensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)

    def test_model_normalizer_float(self):
        model = Normalizer(norm='l2')
        model_onnx = convert_sklearn(model, 'scikit-learn normalizer', [('input', FloatTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
        dump_data_and_model(numpy.array([[1, 1]], dtype=numpy.float32),
                            model, model_onnx, basename="SklearnNormalizerL2-SkipDim1")



if __name__ == "__main__":
    unittest.main()
