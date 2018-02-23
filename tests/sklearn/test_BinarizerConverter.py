"""
Tests scikit-binarizer converter.
"""
import unittest
from sklearn.preprocessing import Binarizer
from onnxmltools.convert.sklearn.BinarizerConverter import BinarizerConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto


class TestSklearnBinarizer(unittest.TestCase):

    def test_model_binarizer(self):
        model = Binarizer(threshold=0.5)
        model_onnx = convert(model, 'scikit-learn binarizer', [('Input', 'float', 1)])
        self.assertTrue(model_onnx is not None)

    def test_binarizer_converter(self):
        model = Binarizer(threshold=0.5)

        context = ConvertContext()
        node = BinarizerConverter.convert(
            context, model, [make_tensor_value_info('input', onnx_proto.TensorProto.FLOAT, [4])])
        self.assertTrue(node is not None)
