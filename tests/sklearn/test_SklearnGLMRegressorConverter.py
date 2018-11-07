"""
Tests GLMRegressor converter.
"""
import unittest
import numpy
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestGLMRegressorConverter(unittest.TestCase):

    def _fit_model(self, model):
        X, y = datasets.make_regression(n_features=4, random_state=0)
        model.fit(X, y)
        return model, X

    def test_model_linear_regression(self):
        model, X = self._fit_model(linear_model.LinearRegression())
        model_onnx = convert_sklearn(model, 'linear regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearRegression-Dec4")

    def test_model_linear_svr(self):
        model, X = self._fit_model(LinearSVR())
        model_onnx = convert_sklearn(model, 'linear SVR', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearSvr-Dec4")

    def test_model_ridge(self):
        model, X = self._fit_model(linear_model.Ridge())
        model_onnx = convert_sklearn(model, 'ridge regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnRidge-Dec4")

    def test_model_sgd_regressor(self):
        model, X = self._fit_model(linear_model.SGDRegressor())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnSGDRegressor-Dec4")

    def test_model_elastic_net_regressor(self):
        model, X = self._fit_model(linear_model.ElasticNet())
        model_onnx = convert_sklearn(model, 'scikit-learn elastic-net regression', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnElasticNet-Dec4")

    def test_model_knn_regressor(self):
        model, X = self._fit_model(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, 'KNN regressor', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        # dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="KNeighborsRegressor")

    def test_model_lasso_lars(self):
        model, X = self._fit_model(linear_model.LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(model, 'lasso lars', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLassoLars-Dec4")


if __name__ == "__main__":
    unittest.main()
