"""
Tests GLMRegressor converter.
"""
import unittest
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


class TestGLMRegressorConverter(unittest.TestCase):

    def _fit_model(self, model):
        X, y = datasets.make_regression(n_features=4, random_state=0)
        model.fit(X, y)
        return model

    def test_model_linear_regression(self):
        model = self._fit_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, 'linear regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svr(self):
        model = self._fit_model(LinearSVR())
        model_onnx = convert_sklearn(model, 'linear SVR', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_ridge(self):
        model = self._fit_model(linear_model.Ridge())
        model_onnx = convert_sklearn(model, 'ridge regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_regressor(self):
        model = self._fit_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_elastic_net_regressor(self):
        model = self._fit_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(model, 'scikit-learn elastic-net regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_knn_regressor(self):
        model = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, 'KNN regressor', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_lasso_lars(self):
        model = self._fit_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(model, 'lasso lars', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
