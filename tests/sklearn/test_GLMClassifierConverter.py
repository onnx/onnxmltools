"""
Tests GLMClassifier converter.
"""
import unittest
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import LinearSVC
from onnxmltools.convert.sklearn.GLMClassifierConverter import GLMClassifierConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.sklearn.SklearnConvertContext import SklearnConvertContext as ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto


class TestGLMClassifierConverter(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model

    def _fit_model_multiclass_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model

    def test_glm_classifier_converter(self):
        model = self._fit_model_binary_classification(linear_model.LogisticRegression())

        context = ConvertContext()
        node = GLMClassifierConverter.convert(
            context, model, [make_tensor_value_info('feature', onnx_proto.TensorProto.FLOAT, [1, 3])])
        self.assertIsNotNone(node)

    def test_model_logistic_regression_binary_class(self):
        model = self._fit_model_binary_classification(linear_model.LogisticRegression())
        model_onnx = convert(model, 'logistic regression', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)

    def test_model_logistic_regression_multi_class(self):
        model = self._fit_model_multiclass_classification(linear_model.LogisticRegression())
        model_onnx = convert(model, 'maximum entropy classifier', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svc_binary_class(self):
        model = self._fit_model_binary_classification(LinearSVC())
        model_onnx = convert(model, 'linear SVC', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svc_multi_class(self):
        model = self._fit_model_multiclass_classification(LinearSVC())
        model_onnx = convert(model, 'multi-class linear SVC', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_binary_class(self):
        model = self._fit_model_binary_classification(linear_model.SGDClassifier())
        model_onnx = convert(model, 'scikit-learn SGD binary classifier', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_multi_class(self):
        model = self._fit_model_multiclass_classification(linear_model.SGDClassifier())
        model_onnx = convert(model, 'scikit-learn SGD multi-class classifier', [('features', 'float', 3)])
        self.assertIsNotNone(model_onnx)
