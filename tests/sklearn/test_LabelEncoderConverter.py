"""
Tests scikit-labebencoder converter.
"""
import unittest
from sklearn.preprocessing import LabelEncoder
from onnxmltools.convert.sklearn.LabelEncoderConverter import LabelEncoderConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto

class TestSklearnLabelEncoderConverter(unittest.TestCase):

    def test_model_label_encoder(self):
        model = LabelEncoder()
        model.fit(['str3', 'str2', 'str0', 'str1', 'str3'])
        model_onnx = convert(model, 'scikit-learn label encoder', [('features','string',1)])
        self.assertTrue(model_onnx is not None)

    def test_label_encoder_converter(self):
        model = LabelEncoder()
        model.fit(['str3', 'str2', 'str0', 'str1', 'str3'])

        context = ConvertContext()
        node = LabelEncoderConverter.convert(context, model,
            [make_tensor_value_info('Input',onnx_proto.TensorProto.STRING, [1])])
        self.assertTrue(node is not None)
