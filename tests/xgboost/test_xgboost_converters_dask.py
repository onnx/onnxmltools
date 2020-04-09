"""
Tests scilit-learn's tree-based methods' converters.
"""
import os
import sys
import unittest
import numpy as np
import pandas
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from onnxruntime import InferenceSession
from xgboost import XGBRegressor, XGBClassifier, train, DMatrix, dask as xgb_dask
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from dask.distributed import Client
from dask_ml.datasets import make_classification


def score_onnx_model(onnx_model, X_test, y_test):
    sess = InferenceSession(onnx_model.SerializeToString())
    pred = sess.run(None, {'X': X_test.astype(np.float32)})[0]
    return mean_squared_error(y_test, pred)
    


class TestXGBoostModelsDask(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2,
                     reason="xgboost converter not tested on python 2")
    def test_xgb_regressor(self):
        client = Client()  # connect to cluster
                

        X, y = make_classification(n_samples=100, n_features=10,
                                   chunks=10, n_informative=4,
                                   random_state=0)
        dtrain = xgb_dask.DaskDMatrix(client, X, y)    
        dask_model = xgb_dask.train(client,
                                {'tree_method': 'hist'},
                                dtrain,
                                num_boost_round=4, evals=[(dtrain, 'train')])        
        onnx_model = convert_xgboost(
            dask_model, initial_types=[('X', FloatTensorType([None, 10]))])

        dump_data_and_model(
            X.astype(np.float32),
            dask_model, model_onnx,
            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
            basename="XGBBoosterDask")


if __name__ == "__main__":
    unittest.main()
