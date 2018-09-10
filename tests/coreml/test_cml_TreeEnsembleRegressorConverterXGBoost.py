"""
Tests CoreML TreeEnsembleRegressor converter.
"""
import os
import unittest
import pandas
from coremltools.converters.xgboost import convert as convert_xgb_to_coreml
from onnxmltools.convert.coreml import convert as convert_cml
from xgboost import XGBRegressor


class TestCoreMLTreeEnsembleRegressorConverterXGBoost(unittest.TestCase):

    def test_tree_ensemble_regressor_xgboost(self):
        
        this = os.path.dirname(__file__)
        data_train = pandas.read_csv(os.path.join(this, "xgboost.model.xgb.n4.d3.train.txt"), header=None)

        X = data_train.iloc[:, 1:].values
        y = data_train.iloc[:, 0].values

        params = dict(n_estimator=4, max_depth=3)
        model = XGBRegressor(**params).fit(X, y)
        # See https://github.com/apple/coremltools/issues/51.
        model.booster = model.get_booster
        model_coreml = convert_xgb_to_coreml(model)
        model_onnx = convert_cml(model_coreml)
        assert model_onnx is not None
