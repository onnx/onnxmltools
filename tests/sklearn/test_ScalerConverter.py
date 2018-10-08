"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
import numpy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnScalerConverter(unittest.TestCase):

    def test_standard_scaler(self):
        model = StandardScaler()
        data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
        model.fit(data)
        model_onnx = convert_sklearn(model, 'scaler', [('input', Int64TensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(numpy.array(data, dtype=numpy.int64),
                            model, model_onnx, basename="StandardScalerInt64")

    def test_standard_scaler_floats(self):
        model = StandardScaler()
        data = [[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]]
        model.fit(data)
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(numpy.array(data, dtype=numpy.float32),
                            model, basename="StandardScalerFloat32")

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

    def test_min_max_scaler(self):
        model = MinMaxScaler()
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

    def test_max_abs_scaler(self):
        model = MaxAbsScaler()
        model.fit([[0., 0., 3.], [1., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        model_onnx = convert_sklearn(model, 'scaler', [('input', FloatTensorType([1, 3]))])
        self.assertTrue(model_onnx is not None)

if __name__ == "__main__":
    unittest.main()
