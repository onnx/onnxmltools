# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from lightgbm import LGBMClassifier
from onnx.defs import onnx_opset_version
from onnxruntime import InferenceSession
from sklearn.datasets import load_breast_cancer

from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.common.onnx_ex import DEFAULT_OPSET_NUMBER

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def _thresholds(tree):
    if "split_feature" in tree:
        yield tree["split_feature"], tree["threshold"]
        yield from _thresholds(tree["left_child"])
        yield from _thresholds(tree["right_child"])


class TestLightGbmFloat32Thresholds(unittest.TestCase):
    def test_float32_threshold_helper(self):
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            _float32_threshold,
        )

        # 868.2000000000002 rounds up to 868.2000122070312 as float32
        t = 868.2000000000002
        t32 = _float32_threshold(t, "<=")
        self.assertLessEqual(t32, t)
        self.assertEqual(np.float32(t32), t32)
        self.assertGreater(float(np.nextafter(np.float32(t32), np.float32(np.inf))), t)
        # a threshold already representable as float32 does not change
        self.assertEqual(_float32_threshold(0.5, "<="), 0.5)
        # other criteria and non finite values are left alone
        self.assertEqual(_float32_threshold(t, "=="), t)
        self.assertTrue(np.isinf(_float32_threshold(np.inf, "<=")))

    def test_inputs_next_to_thresholds_follow_lightgbm(self):
        X, y = load_breast_cancer(return_X_y=True)
        model = LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=0, verbose=-1
        ).fit(X, y)
        onx = convert_lightgbm(
            model,
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        base = X.mean(axis=0).astype(np.float32)
        rows = []
        for tree in model.booster_.dump_model()["tree_info"]:
            for feature, threshold in _thresholds(tree["tree_structure"]):
                t32 = np.float32(threshold)
                for value in (
                    np.nextafter(t32, np.float32(-np.inf)),
                    t32,
                    np.nextafter(t32, np.float32(np.inf)),
                ):
                    row = base.copy()
                    row[feature] = value
                    rows.append(row)
        rows = np.array(rows, dtype=np.float32)
        expected = model.predict_proba(rows.astype(np.float64))
        labels, probas = sess.run(None, {"X": rows})
        np.testing.assert_allclose(probas, expected, atol=1e-6)
        np.testing.assert_array_equal(labels, model.predict(rows.astype(np.float64)))

    def test_rounded_decimal_input(self):
        # 868.2 is a float32 value just above the float64 threshold
        # 868.2000000000002 used by this model.
        X, y = load_breast_cancer(return_X_y=True)
        model = LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=0, verbose=-1
        ).fit(X, y)
        onx = convert_lightgbm(
            model,
            initial_types=[("X", FloatTensorType([None, X.shape[1]]))],
            zipmap=False,
            target_opset=TARGET_OPSET,
        )
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        found = False
        for tree in model.booster_.dump_model()["tree_info"]:
            for feature, threshold in _thresholds(tree["tree_structure"]):
                if abs(threshold - 868.2000000000002) < 1e-9:
                    row = X.mean(axis=0).astype(np.float32)
                    row[feature] = np.float32(868.2)
                    found = True
                    label_ref = model.predict(row[None, :].astype(np.float64))
                    label_onnx = sess.run(None, {"X": row[None, :]})[0]
                    np.testing.assert_array_equal(label_onnx, label_ref)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
