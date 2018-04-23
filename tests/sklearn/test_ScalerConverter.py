"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
from sklearn.preprocessing import StandardScaler
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType

class TestSklearnScalerConverter(unittest.TestCase):

    def test_model_scaler(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        model_onnx = convert_sklearn(model, 'scaler', [Int64TensorType([1, 3])])
        self.assertTrue(model_onnx is not None)

    def test_scaler_converter_floats(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        model_onnx = convert_sklearn(model, 'scaler', [FloatTensorType([1, 3])])
        self.assertTrue(model_onnx is not None)

