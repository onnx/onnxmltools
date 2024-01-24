# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import scipy
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from xgboost import XGBClassifier, XGBRegressor
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestXGBoostModelsBaseScore(unittest.TestCase):
    def test_xgbregressor_sparse_base_score(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=0)
        mask = np.random.randint(0, 50, size=(X.shape)) != 0
        X[mask] = 0
        y = (y + mask.sum(axis=1, keepdims=0)).astype(np.float32)
        X_sp = scipy.sparse.coo_matrix(X)
        X = X.astype(np.float32)

        rf = XGBRegressor(n_estimators=3, max_depth=4, random_state=0, base_score=0.5)
        rf.fit(X_sp, y)
        expected = rf.predict(X).astype(np.float32).reshape((-1, 1))
        expected_sparse = rf.predict(X_sp).astype(np.float32).reshape((-1, 1))
        diff = np.abs(expected - expected_sparse)
        self.assertNotEqual(diff.min(), diff.max())

        onx = convert_xgboost(
            rf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        feeds = {"X": X}

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        assert_almost_equal(expected, got, decimal=4)

    def test_xgbregressor_sparse_no_base_score(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=0)
        mask = np.random.randint(0, 50, size=(X.shape)) != 0
        X[mask] = 0
        y = (y + mask.sum(axis=1, keepdims=0)).astype(np.float32)
        X_sp = scipy.sparse.coo_matrix(X)
        X = X.astype(np.float32)

        rf = XGBRegressor(n_estimators=3, max_depth=4, random_state=0)
        rf.fit(X_sp, y)
        expected = rf.predict(X).astype(np.float32).reshape((-1, 1))
        expected_sparse = rf.predict(X_sp).astype(np.float32).reshape((-1, 1))
        diff = np.abs(expected - expected_sparse)
        self.assertNotEqual(diff.min(), diff.max())

        onx = convert_xgboost(
            rf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        feeds = {"X": X}

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        assert_almost_equal(expected, got, decimal=4)

    def test_xgbclassifier_sparse_base_score(self):
        X, y = make_regression(n_samples=200, n_features=10, random_state=0)
        mask = np.random.randint(0, 50, size=(X.shape)) != 0
        X[mask] = 0
        y = (y + mask.sum(axis=1, keepdims=0)).astype(np.float32)
        y = y >= y.mean()
        X_sp = scipy.sparse.coo_matrix(X)
        X = X.astype(np.float32)

        rf = XGBClassifier(n_estimators=3, max_depth=4, random_state=0, base_score=0.5)
        rf.fit(X_sp, y)
        expected = rf.predict_proba(X).astype(np.float32).reshape((-1, 1))
        expected_sparse = rf.predict_proba(X_sp).astype(np.float32).reshape((-1, 1))
        diff = np.abs(expected - expected_sparse)
        self.assertNotEqual(diff.min(), diff.max())

        onx = convert_xgboost(
            rf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        feeds = {"X": X}

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[1]
        assert_almost_equal(expected.reshape((-1, 2)), got, decimal=4)

    def test_xgbclassifier_sparse_no_base_score(self):
        X, y = make_regression(n_samples=400, n_features=10, random_state=0)
        mask = np.random.randint(0, 50, size=(X.shape)) != 0
        X[mask] = 0
        y = (y + mask.sum(axis=1, keepdims=0)).astype(np.float32)
        y = y >= y.mean()
        X_sp = scipy.sparse.coo_matrix(X)
        X = X.astype(np.float32)

        rf = XGBClassifier(n_estimators=3, max_depth=4, random_state=0)
        rf.fit(X_sp, y)
        expected = rf.predict_proba(X).astype(np.float32).reshape((-1, 1))
        expected_sparse = rf.predict_proba(X_sp).astype(np.float32).reshape((-1, 1))
        diff = np.abs(expected - expected_sparse)
        self.assertNotEqual(diff.min(), diff.max())

        onx = convert_xgboost(
            rf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        feeds = {"X": X}

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[1]
        assert_almost_equal(expected.reshape((-1, 2)), got, decimal=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
