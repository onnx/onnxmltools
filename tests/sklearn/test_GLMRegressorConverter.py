"""
Tests GLMRegressor converter.
"""
import unittest
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import LinearSVR
from onnxmltools.convert.sklearn.GLMRegressorConverter import GLMRegressorConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto


class TestGLMRegressorConverter(unittest.TestCase):

    def _fit_model(self, model):
        X, y = datasets.make_regression(n_features=4, random_state=0)
        model.fit(X, y)
        return model

    def test_glm_regressor_converter(self):
        model = self._fit_model(linear_model.LinearRegression())

        context = ConvertContext()
        node = GLMRegressorConverter.convert(
            context, model, [make_tensor_value_info('feature', onnx_proto.TensorProto.FLOAT, [1, 4])])
        self.assertIsNotNone(node)

    def test_model_linear_regression(self):
        model = self._fit_model(linear_model.LinearRegression())
        model_onnx = convert(model, 'linear regression', [('features', 'double', 4)])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svr(self):
        model = self._fit_model(LinearSVR())
        model_onnx = convert(model, 'linear SVR', [('features', 'double', 4)])
        self.assertIsNotNone(model_onnx)

    def test_model_ridge(self):
        model = self._fit_model(linear_model.Ridge())
        model_onnx = convert(model, 'ridge regression', [('features', 'double', 4)])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_regressor(self):
        model = self._fit_model(linear_model.SGDRegressor())
        model_onnx = convert(model, 'scikit-learn SGD regression', [('features', 'double', 4)])
        self.assertIsNotNone(model_onnx)
