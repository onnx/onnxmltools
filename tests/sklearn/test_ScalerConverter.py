"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
from sklearn.preprocessing import StandardScaler, RobustScaler
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType

class TestSklearnScalerConverter(unittest.TestCase):

    def test_standard_scaler(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', Int64TensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_standard_scaler_floats(self):
        model = StandardScaler()
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_robust_scaler_floats(self):
        model = RobustScaler()
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_robust_scaler_floats_no_bias(self):
        model = RobustScaler(with_centering=False)
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_robust_scaler_floats_no_scaling(self):
        model = RobustScaler(with_scaling=False)
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)
