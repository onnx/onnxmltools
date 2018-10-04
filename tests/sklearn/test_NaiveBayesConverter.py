import unittest
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


class TestNaiveBayesConverter(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        data = load_iris()
        X = data.data
        y = data.target
        y[y == 2] = 1
        model.fit(X, y)
        return model 

    def _fit_model_multiclass_classification(self, model):
        data = load_iris()
        X = data.data
        y = data.target
        model.fit(X, y)
        return model

    def test_model_multinomial_nb_binary_classification(self):
        model = self._fit_model_binary_classification(MultinomialNB())
        model_onnx = convert_sklearn(model, 'multinomial naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_bernoulli_nb_binary_classification(self):
        model = self._fit_model_binary_classification(BernoulliNB())
        model_onnx = convert_sklearn(model, 'bernoulli naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_multinomial_nb_multiclass(self):
        model = self._fit_model_multiclass_classification(MultinomialNB())
        model_onnx = convert_sklearn(model, 'multinomial naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)

    def test_model_bernoulli_nb_multiclass(self):
        model = self._fit_model_multiclass_classification(BernoulliNB())
        model_onnx = convert_sklearn(model, 'bernoulli naive bayes', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
