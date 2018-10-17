import unittest
from sklearn import datasets
from lightgbm import LGBMClassifier

from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType

class TestLGBMClassifierConverter(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model

    def test_model_binary_classification(self):
        model = self._fit_model_binary_classification(LGBMClassifier(objective="binary", boosting_type="gbdt"))
        model_onnx = convert_sklearn(model, 'scikit-learn LGBM binary classifier', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)