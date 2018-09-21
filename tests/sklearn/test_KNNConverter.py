import unittest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_iris
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType


class TestKNNConverter(unittest.TestCase):

    def _fit_model_multiclass_classification(self, model):
        data = load_iris()
        X = data.data
        y = data.target
        model.fit(X, y)
        return model

    def test_model_knn_regressor(self):
        model = self._fit_model_multiclass_classification(KNeighborsRegressor(n_neighbors=2))
        model_onnx = convert_sklearn(model, 'KNN regressor', [('input', FloatTensorType([1, 4]))])
        self.assertIsNotNone(model_onnx)
