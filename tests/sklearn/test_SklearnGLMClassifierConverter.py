import unittest
import numpy
from sklearn import datasets
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestGLMClassifierConverter(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X

    def _fit_model_multiclass_classification(self, model):
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model, X

    def test_model_logistic_regression_binary_class(self):
        model, X = self._fit_model_binary_classification(linear_model.LogisticRegression())
        model_onnx = convert_sklearn(model, 'logistic regression', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLogitisticRegressionBinary")

    def test_model_logistic_regression_multi_class(self):
        model, X = self._fit_model_multiclass_classification(linear_model.LogisticRegression())
        model_onnx = convert_sklearn(model, 'maximum entropy classifier', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLogitisticRegressionMulti")

    def test_model_linear_svc_binary_class(self):
        model, X = self._fit_model_binary_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'linear SVC', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearSVCBinary-NoProb")

    def test_model_linear_svc_multi_class(self):
        model, X = self._fit_model_multiclass_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'multi-class linear SVC', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearSVCMulti")

    def test_model_sgd_binary_class(self):
        model, X = self._fit_model_binary_classification(linear_model.SGDClassifier())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD binary classifier', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnSGDClassifierBinary-NoProb-Dec4")

    def test_model_sgd_multi_class(self):
        model, X = self._fit_model_multiclass_classification(linear_model.SGDClassifier())
        model_onnx = convert_sklearn(model, 'scikit-learn SGD multi-class classifier',
                                     [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnSGDClassifierMulti-Dec3")

    def test_model_knn_classifier_binary_class(self):
        model, X = self._fit_model_binary_classification(KNeighborsClassifier())
        model_onnx = convert_sklearn(model, 'KNN classifier binary', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(numpy.atleast_2d(X[0]).astype(numpy.float32), model, model_onnx,
                            basename="SklearnKNeighborsClassifierBinary",
                            allow_failure="StrictVersion(onnx.__version__) == StrictVersion('1.1.2')")

    def test_model_knn_classifier_multi_class(self):
        model, X = self._fit_model_multiclass_classification(KNeighborsClassifier())
        model_onnx = convert_sklearn(model, 'KNN classifier multi-class', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(numpy.atleast_2d(X[0]).astype(numpy.float32), model, model_onnx,
                            basename="SklearnKNeighborsClassifierMulti",
                            allow_failure="StrictVersion(onnx.__version__) == StrictVersion('1.1.2')")

if __name__ == "__main__":
    # TestGLMClassifierConverter().test_model_linear_svc_multi_class()
    unittest.main()
