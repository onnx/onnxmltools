"""
Tests scikit-normalizer converter.
"""
import unittest
from sklearn.preprocessing import Normalizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.coreml._data_types import Int64TensorType

class TestSklearnNormalizerConverter(unittest.TestCase):

    def test_model_normalizer(self):
        model = Normalizer(norm='l2')
        model_onnx = convert_sklearn(model, 'scikit-learn normalizer', [Int64TensorType([1, 1])])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(len(model_onnx.graph.node) == 1)
