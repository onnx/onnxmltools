import unittest
from sklearn import datasets
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

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

    def _fit_model_multiclass_classification(self, model):
        digits = datasets.load_digits()
        X = digits.data[:, :10]
        y = digits.target
        model.fit(X, y)
        return model

    def _fit_model_regression(self, model):
        diabetes = datasets.load_digits()
        X = diabetes.data
        y = diabetes.target
        model.fit(X, y)
        return model

    def test_model_binary_classification(self):
        model = self._fit_model_binary_classification(LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            boosting_type="gbdt"))
        model_onnx = convert_sklearn(model, 'scikit-learn LGBM binary classifier', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)

    def test_model_multiclass_classification(self):
        model = self._fit_model_binary_classification(LGBMClassifier(
            objective="ova", 
            learning_rate=0.05, 
            boosting_type="gbdt", 
            num_class=10))
        model_onnx = convert_sklearn(model, 'scikit-learn LGBM multiclass classifier', [('input', FloatTensorType([1, 10]))])
        self.assertIsNotNone(model_onnx)

    def test_model_regression(self):
        model = self._fit_model_regression(LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            metric="rmsle"))
        model_onnx = convert_sklearn(model, 'scikit-learn LGBM regression', [('input', FloatTensorType([1, 10]))])
        self.assertIsNotNone(model_onnx)