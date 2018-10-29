import unittest
import numpy
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestNaiveBayesConverter(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        data = load_iris()
        X = data.data
        y = data.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X.astype(numpy.float32)

    def _fit_model_multiclass_classification(self, model):
        data = load_iris()
        X = data.data
        y = data.target
        model.fit(X, y)
        return model, X.astype(numpy.float32)

    def test_model_multinomial_nb_binary_classification(self):
        model, X = self._fit_model_binary_classification(MultinomialNB())
        model_onnx = convert_sklearn(model, 'multinomial naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnBinMultinomialNB-OneOff",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")

    def test_model_bernoulli_nb_binary_classification(self):
        model, X = self._fit_model_binary_classification(BernoulliNB())
        model_onnx = convert_sklearn(model, 'bernoulli naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X[:5], model, model_onnx, basename="SklearnBinBernoulliNB-OneOff",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.4')")

    def test_model_multinomial_nb_multiclass(self):
        model, X = self._fit_model_multiclass_classification(MultinomialNB())
        model_onnx = convert_sklearn(model, 'multinomial naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclMultinomialNB-OneOff",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")

    def test_model_bernoulli_nb_multiclass(self):
        model, X = self._fit_model_multiclass_classification(BernoulliNB())
        model_onnx = convert_sklearn(model, 'bernoulli naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclBernoulliNB-OneOff",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")


if __name__ == "__main__":
    unittest.main()
