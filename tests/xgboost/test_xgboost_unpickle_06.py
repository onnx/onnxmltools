"""
Tests scikit-linear converter.
"""
import os
from xgboost import XGBRegressor, XGBClassifier
from onnxmltools.convert.xgboost import convert as convert_xgboost
from sklearn.model_selection import train_test_split
import pandas
import numpy
import unittest
import pickle
from onnxmltools.convert.common.data_types import FloatTensorType

class TestXGBoostUnpickle06(unittest.TestCase):
    
    def test_xgboost_unpickle_06(self):
        # Unpickle a model trained with an old version of xgboost.
        this = os.path.dirname(__file__)
        with open(os.path.join(this, "xgboost10day.pickle.dat"), "rb") as f:
            xgb = pickle.load(f)
        
        conv_model = convert_xgboost(xgb, initial_types=[('features', FloatTensorType([1, 10000]))])
        assert conv_model is not None


if __name__ == "__main__":
    unittest.main()

