"""
Tests scikit-linear converter.
"""
import sys
import os
from onnxmltools.convert.xgboost import convert as convert_xgboost
import unittest
import pickle
from onnxmltools.convert.common.data_types import FloatTensorType


class TestXGBoostUnpickle06(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2, reason="xgboost converted not tested on python 2")
    def test_xgboost_unpickle_06(self):
        # Unpickle a model trained with an old version of xgboost.
        this = os.path.dirname(__file__)
        with open(os.path.join(this, "xgboost10day.pickle.dat"), "rb") as f:
            xgb = pickle.load(f)

        conv_model = convert_xgboost(xgb, initial_types=[('features', FloatTensorType(['None', 10000]))])
        assert conv_model is not None


if __name__ == "__main__":
    unittest.main()
