"""
Tests scikit-onehotencoder converter.
"""
import unittest
from sklearn.preprocessing import OneHotEncoder
from onnxmltools.convert.sklearn.convert import convert

class TestSklearnOneHotEncoderConverter(unittest.TestCase):

    def test_model_one_hot_encoder(self):
        model = OneHotEncoder()
        model.fit([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]])
        model_onnx = convert(model, 'scikit-learn one-hot encoder', [('features', 'int64', 3)])
        self.assertTrue(model_onnx is not None)

    def test_one_hot_encoder_mixed_float_int(self):
        model = OneHotEncoder()
        model.fit([[0.4, 0.2, 3], [1.4, 1.2, 0], [0.2, 2.2, 1]])
        node = convert(model, 'test', [('features', 'float', 2),
                                       ('features2', 'int64', 1)])
        self.assertTrue(node is not None)
