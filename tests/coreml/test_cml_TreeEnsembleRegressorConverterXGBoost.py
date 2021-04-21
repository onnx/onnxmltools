# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML TreeEnsembleRegressor converter.
"""
import os
import sys
import unittest
import numpy
import pandas
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
from coremltools.converters.xgboost import convert as convert_xgb_to_coreml
from onnxmltools.convert.coreml import convert as convert_cml
from xgboost import XGBRegressor
from onnxmltools.utils import dump_data_and_model


class TestCoreMLTreeEnsembleRegressorConverterXGBoost(unittest.TestCase):

    @unittest.skipIf(True, reason="broken")
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
        if sys.version_info[0] >= 3:
            # python 2.7 returns TypeError: can't pickle instancemethod objects
            dump_data_and_model(X.astype(numpy.float32), model, model_onnx,
                                         basename="CmlXGBoostRegressor-OneOff-Reshape",
                                         allow_failure=True)


if __name__ == "__main__":
    unittest.main()
