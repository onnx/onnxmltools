"""
Tests scikit-labebencoder converter.
"""
import unittest
import numpy
from sklearn.preprocessing import LabelEncoder
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import StringTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnLabelEncoderConverter(unittest.TestCase):

    def test_model_label_encoder(self):
        model = LabelEncoder()
        data = ['str3', 'str2', 'str0', 'str1', 'str3']
        model.fit(data)
        model_onnx = convert_sklearn(model, 'scikit-learn label encoder', [('input', StringTensorType([1, 1]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        dump_data_and_model(numpy.array(data),
                            model, model_onnx, basename="SklearnLabelEncoder")



if __name__ == "__main__":
    unittest.main()
