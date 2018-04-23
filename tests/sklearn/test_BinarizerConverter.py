import unittest
from sklearn.preprocessing import Binarizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


class TestSklearnBinarizer(unittest.TestCase):

    def test_model_binarizer(self):
        model = Binarizer(threshold=0.5)
        model_onnx = convert_sklearn(model, 'scikit-learn binarizer', [FloatTensorType([1, 1])])
        self.assertTrue(model_onnx is not None)

