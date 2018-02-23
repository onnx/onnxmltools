"""
Tests scikit-learn's standard scaler converter.
"""
import unittest
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert.sklearn.ScalerConverter import ScalerConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto

class TestSklearnScalerConverter(unittest.TestCase):

    def test_model_scaler(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        model_onnx = convert(model, 'scaler', [('features','int64', 3)])
        self.assertTrue(model_onnx is not None)

    def test_scaler_converter(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

        context = ConvertContext()
        model_onnx = ScalerConverter.convert(context, model,
            [make_tensor_value_info('features', onnx_proto.TensorProto.INT64, [3])])
        self.assertTrue(model_onnx is not None)
