"""
Tests scikit-normalizer converter.
"""
import unittest
from sklearn.preprocessing import Normalizer
from onnxmltools.convert.sklearn.NormalizerConverter import NormalizerConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto

class TestSklearnNormalizerConverter(unittest.TestCase):

    def test_model_normalizer(self):
        model = Normalizer(norm='l2')
        model_onnx = convert(model, 'scikit-learn normalizer', [('features','int64',1)])
        self.assertTrue(model_onnx is not None)
 
    def test_normalizer_converter(self):
        model = Normalizer(norm='l2')

        context = ConvertContext()
        node = NormalizerConverter.convert(context, model,
            [make_tensor_value_info('feature1', onnx_proto.TensorProto.INT64, [1])])
        self.assertTrue(node is not None)

