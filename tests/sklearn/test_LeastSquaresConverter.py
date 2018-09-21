import unittest
from sklearn.linear_model import LassoLars, Ridge 
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


class TestLeastSquaresConverter(unittest.TestCase):

    def _fit_model_least_squares(self, model):
        X = [[-1, 1], [0, 0], [1, 1]]
        y = [-1, 0, -1] 
        model.fit(X, y)
        return model 

    def test_model_lasso_lars(self):
        model = self._fit_model_least_squares(LassoLars(alpha=0.01))
        model_onnx = convert_sklearn(model, 'lasso lars', [('input', FloatTensorType([1, 2]))])
        self.assertIsNotNone(model_onnx)

    def test_model_ridge(self):
        model = self._fit_model_least_squares(Ridge())
        model_onnx = convert_sklearn(model, 'ridge', [('input', FloatTensorType([1, 2]))])
        self.assertIsNotNone(model_onnx)
