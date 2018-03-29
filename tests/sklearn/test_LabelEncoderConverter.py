"""
Tests scikit-labebencoder converter.
"""
import unittest
from sklearn.preprocessing import LabelEncoder
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import StringTensorType

class TestSklearnLabelEncoderConverter(unittest.TestCase):

    def test_model_label_encoder(self):
        model = LabelEncoder()
        model.fit(['str3', 'str2', 'str0', 'str1', 'str3'])
        model_onnx = convert_sklearn(model, 'scikit-learn label encoder', [StringTensorType([1, 1])])
        self.assertTrue(model_onnx is not None)

    def test_label_encoder_converter(self):
        model = LabelEncoder()
        model.fit(['str3', 'str2', 'str0', 'str1', 'str3'])

        model_onnx = convert_sklearn(model, 'scikit-learn label encoder', [StringTensorType([1, 1])])
        self.assertTrue(model_onnx.graph.node is not None)
