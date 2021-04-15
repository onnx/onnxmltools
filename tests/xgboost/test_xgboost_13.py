# SPDX-License-Identifier: Apache-2.0

"""
Tests scilit-learn's tree-based methods' converters.
"""
import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier, train, DMatrix
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


class TestXGBoost13(unittest.TestCase):

    def test_xgb_regressor(self):
        this = os.path.dirname(__file__)
        df = pandas.read_csv(os.path.join(this, "data_fail_empty.csv"))
        X, y = df.drop('y', axis=1), df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clr = XGBClassifier(
            max_delta_step= 0, tree_method='hist', n_estimators=100,
            booster='gbtree', objective='binary:logistic', eval_metric='logloss',
            learning_rate= 0.1, gamma=10, max_depth=7, min_child_weight=50,
            subsample=0.75, colsample_bytree=0.75, random_state=42,
            verbosity=0)

        clr.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                early_stopping_rounds=40)

        initial_type = [('float_input', FloatTensorType([None, 797]))]
        onx = convert_xgboost(clr, initial_types=initial_type)
        expected = clr.predict(X_test), clr.predict_proba(X_test)
        sess = InferenceSession(onx.SerializeToString())
        X_test = X_test.values.astype(np.float32)
        got = sess.run(None, {'float_input': X_test})
        assert_almost_equal(expected[1], got[1])
        assert_almost_equal(expected[0], got[0])


if __name__ == "__main__":
    unittest.main()
