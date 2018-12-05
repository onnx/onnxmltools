"""
Tests scilit-learn's tree-based methods' converters.
"""
import unittest
from sklearn.datasets import load_iris
from xgboost import XGBRegressor, XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


class TestXGBoostModels(unittest.TestCase):
    
    def test_xgb_regressor(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBRegressor()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)
        
    def test_xgb_classifier(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 0

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)

    def test_xgb_classifier_multi(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        xgb = XGBClassifier()
        xgb.fit(X, y)
        conv_model = convert_xgboost(xgb, initial_types=[('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(conv_model is not None)


if __name__ == "__main__":
    unittest.main()
