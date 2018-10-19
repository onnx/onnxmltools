"""
Tests CoreML TreeEnsembleRegressor converter.
"""
import unittest
import numpy
import coremltools
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLTreeEnsembleRegressorConverter(unittest.TestCase):

    def test_tree_ensemble_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)
        model = RandomForestRegressor().fit(X, y)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="CmlRegRandomForestRegressor-Dec4")


if __name__ == "__main__":
    unittest.main()
