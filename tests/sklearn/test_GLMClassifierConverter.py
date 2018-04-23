import unittest
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import LinearSVC
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


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

    def test_model_logistic_regression_binary_class(self):
        model = self._fit_model_binary_classification(linear_model.LogisticRegression())
        model_onnx = convert_sklearn(model, 'logistic regression', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)

    def test_model_logistic_regression_multi_class(self):
        model = self._fit_model_multiclass_classification(linear_model.LogisticRegression())
        model_onnx = convert_sklearn(model, 'maximum entropy classifier', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svc_binary_class(self):
        model = self._fit_model_binary_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'linear SVC', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)

    def test_model_linear_svc_multi_class(self):
        model = self._fit_model_multiclass_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'multi-class linear SVC', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_binary_class(self):
        model = self._fit_model_binary_classification(linear_model.SGDClassifier())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD binary classifier', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)

    def test_model_sgd_multi_class(self):
        model = self._fit_model_multiclass_classification(linear_model.SGDClassifier())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD multi-class classifier', [FloatTensorType([1, 3])])
        self.assertIsNotNone(model_onnx)
