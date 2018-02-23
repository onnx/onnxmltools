"""
Tests CoreML TreeEnsembleRegressor converter.
"""
import coremltools
import unittest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLTreeEnsembleRegressorConverter(unittest.TestCase):

    def test_tree_ensemble_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)
        model = RandomForestRegressor().fit(X, y)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)