# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy
from numpy.testing import assert_array_equal
from onnx.defs import onnx_opset_version
from lightgbm import LGBMClassifier, LGBMRegressor
import onnxruntime
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_lightgbm

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestDecisionLeaf(unittest.TestCase):
    def _make_data(self):
        X = numpy.array(
            [[0, 1], [1, 1], [2, 0], [1, 2], [1, 5], [6, 2]], dtype=numpy.float32
        )
        y_bin = [0, 1, 0, 1, 1, 0]
        y_reg = [0.1, 1.2, 0.3, 1.4, 1.5, 0.6]
        return X, y_bin, y_reg

    def test_decision_leaf_binary_classifier(self):
        X, y, _ = self._make_data()
        model = LGBMClassifier(
            n_estimators=3, min_child_samples=1, max_depth=2, num_thread=1
        )
        model.fit(X, y)
        onx = convert_lightgbm(
            model,
            "dummy",
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
            decision_leaf=True,
        )
        # Model must have 3 outputs: label, probabilities, leaf_indices
        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 3)

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, {"X": X})
        self.assertEqual(len(results), 3)

        label, proba, leaf_indices = results
        # leaf_indices shape: [N, n_trees]
        self.assertEqual(leaf_indices.shape, (X.shape[0], model.n_estimators_))
        self.assertEqual(leaf_indices.dtype, numpy.int64)

        # leaf_indices values should be non-negative integers (node IDs in the tree)
        self.assertTrue((leaf_indices >= 0).all())
        # Each tree should produce consistent indices (same input → same leaf)
        leaf_indices2 = sess.run(None, {"X": X})[2]
        assert_array_equal(leaf_indices, leaf_indices2)

    def test_decision_leaf_multiclass_classifier(self):
        X = numpy.array(
            [[0, 1], [1, 1], [2, 0], [1, 2], [0, 3], [3, 0]], dtype=numpy.float32
        )
        y = [0, 1, 2, 0, 1, 2]
        model = LGBMClassifier(
            n_estimators=3, min_child_samples=1, num_class=3, num_thread=1
        )
        model.fit(X, y)
        n_trees = model.booster_.num_trees()
        onx = convert_lightgbm(
            model,
            "dummy",
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
            decision_leaf=True,
        )
        self.assertEqual(len(onx.graph.output), 3)

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, {"X": X})
        self.assertEqual(len(results), 3)

        _, _, leaf_indices = results
        self.assertEqual(leaf_indices.shape, (X.shape[0], n_trees))
        self.assertEqual(leaf_indices.dtype, numpy.int64)
        self.assertTrue((leaf_indices >= 0).all())

    def test_decision_leaf_regressor(self):
        X, _, y = self._make_data()
        model = LGBMRegressor(
            n_estimators=3, min_child_samples=1, max_depth=2, num_thread=1
        )
        model.fit(X, y)
        n_trees = model.booster_.num_trees()
        onx = convert_lightgbm(
            model,
            "dummy",
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET,
            decision_leaf=True,
        )
        # Model must have 2 outputs: prediction, leaf_indices
        self.assertEqual(len(onx.graph.output), 2)

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, {"X": X})
        self.assertEqual(len(results), 2)

        pred, leaf_indices = results
        self.assertEqual(leaf_indices.shape, (X.shape[0], n_trees))
        self.assertEqual(leaf_indices.dtype, numpy.int64)
        self.assertTrue((leaf_indices >= 0).all())

    def test_decision_leaf_false_keeps_original_outputs(self):
        """Ensure decision_leaf=False (default) produces the original outputs."""
        X, y, _ = self._make_data()
        model = LGBMClassifier(
            n_estimators=3, min_child_samples=1, max_depth=2, num_thread=1
        )
        model.fit(X, y)
        onx = convert_lightgbm(
            model,
            "dummy",
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
            decision_leaf=False,
        )
        self.assertEqual(len(onx.graph.output), 2)

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        results = sess.run(None, {"X": X})
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
