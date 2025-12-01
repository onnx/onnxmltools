# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import scipy
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression, make_classification
import json

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBRegressor = None
from onnx.defs import onnx_opset_version
from onnxmltools.convert.common.onnx_ex import DEFAULT_OPSET_NUMBER

if XGBRegressor is not None:
    from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestXGBoostModelsBaseScore(unittest.TestCase):
    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
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
        # expected sparse is expected ot be diffrent than expected,
        # expected_sparse = rf.predict(X_sp).astype(np.float32).reshape((-1, 1))

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

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
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
        # expected sparse is expected ot be diffrent than expected,
        # expected_sparse = rf.predict(X_sp).astype(np.float32).reshape((-1, 1))

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

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
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
        # expected sparse is expected ot be diffrent than expected,
        # expected_sparse = rf.predict_proba(X_sp).astype(np.float32).reshape((-1, 1))

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

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
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
        # expected sparse is expected ot be diffrent than expected,
        # expected_sparse = rf.predict_proba(X_sp).astype(np.float32).reshape((-1, 1))

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

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_xgbclassifier_multiclass_base_score(self):
        """Test multiclass classifier - xgboost 3 can have different base_scores per class"""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3,
            n_informative=5, n_redundant=0, random_state=42
        )
        X = X.astype(np.float32)

        clf = XGBClassifier(n_estimators=3, max_depth=4, random_state=42)
        clf.fit(X, y)
        expected = clf.predict_proba(X).astype(np.float32)

        onx = convert_xgboost(
            clf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        feeds = {"X": X}

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[1]
        assert_almost_equal(expected, got, decimal=4)

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_xgbclassifier_multiclass_base_score_in_onnx(self):
        """Verify that base_values are actually present in the ONNX graph"""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=3,
            n_informative=5, n_redundant=0, random_state=42
        )
        X = X.astype(np.float32)

        clf = XGBClassifier(n_estimators=3, max_depth=4, random_state=42)
        clf.fit(X, y)

        config = json.loads(clf.get_booster().save_config())
        base_score_str = config["learner"]["learner_model_param"]["base_score"]

        onx = convert_xgboost(
            clf,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        tree_ensemble_node = None
        for node in onx.graph.node:
            if node.op_type == "TreeEnsembleClassifier":
                tree_ensemble_node = node
                break

        self.assertIsNotNone(tree_ensemble_node, "TreeEnsembleClassifier node not found")

        base_values = None
        for attr in tree_ensemble_node.attribute:
            if attr.name == "base_values":
                base_values = list(attr.floats)
                break

        self.assertIsNotNone(base_values, "base_values attribute not found in ONNX model")
        self.assertEqual(len(base_values), 3, "base_values should have 3 elements for 3-class problem")

        # In xgboost 3+, base_score is a string array like "[3.4E-1,3.3E-1,3.3E-1]"
        # Verify that base_values in ONNX match the xgboost config
        if base_score_str.startswith("[") and base_score_str.endswith("]"):
            expected_base_scores = json.loads(base_score_str)
            for i, val in enumerate(base_values):
                if i < len(expected_base_scores):
                    self.assertAlmostEqual(val, expected_base_scores[i], places=5)

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_xgbregressor_base_score_in_onnx(self):
        """Verify that regressor base_values are present in the ONNX graph"""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        reg = XGBRegressor(n_estimators=3, max_depth=4, random_state=42)
        reg.fit(X, y)

        onx = convert_xgboost(
            reg,
            initial_types=[("X", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )

        tree_ensemble_node = None
        for node in onx.graph.node:
            if node.op_type == "TreeEnsembleRegressor":
                tree_ensemble_node = node
                break

        self.assertIsNotNone(tree_ensemble_node, "TreeEnsembleRegressor node not found")

        base_values = None
        for attr in tree_ensemble_node.attribute:
            if attr.name == "base_values":
                base_values = list(attr.floats)
                break

        self.assertIsNotNone(base_values, "base_values attribute not found in ONNX model")
        self.assertGreater(len(base_values), 0, "base_values should not be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
