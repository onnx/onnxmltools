# SPDX-License-Identifier: Apache-2.0

import unittest
import packaging.version as pv
import lightgbm
import numpy
from numpy.testing import assert_almost_equal
from onnx.defs import onnx_opset_version
from lightgbm import LGBMRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from onnxruntime import InferenceSession, __version__ as ort_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
ort_version = ".".join(ort_version.split(".")[:2])


class TestLightGbmTreeEnsembleModelsSplit(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.7.0"),
        reason="Sum<double> not implemented.",
    )
    def test_lgbm_regressor10(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        reg = LGBMRegressor(max_depth=2, n_estimators=4, seed=0, num_thread=1)
        reg.fit(X_train, y_train)
        expected = reg.predict(X_test)

        # float
        init = [("X", FloatTensorType([None, X_train.shape[1]]))]
        onx = convert_lightgbm(reg, None, init, target_opset=TARGET_OPSET)
        self.assertNotIn('op_type: "Sum"', str(onx))
        oinf = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got1 = oinf.run(None, {"X": X_test})[0]

        # float split
        onx = convert_lightgbm(reg, None, init, split=2, target_opset=TARGET_OPSET)
        self.assertIn('op_type: "Sum"', str(onx))
        oinf = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got2 = oinf.run(None, {"X": X_test})[0]

        # final check
        assert_almost_equal(expected, got1.ravel(), decimal=5)
        assert_almost_equal(expected, got2.ravel(), decimal=5)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.7.0"),
        reason="Sum<double> not implemented.",
    )
    def test_lgbm_regressor(self):
        data = load_iris()
        X, y = data.data, data.target
        X = X.astype(numpy.float32)
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        reg = LGBMRegressor(max_depth=2, n_estimators=100, seed=0, num_thread=1)
        reg.fit(X_train, y_train)
        expected = reg.predict(X_test)

        # float
        init = [("X", FloatTensorType([None, X_train.shape[1]]))]
        onx = convert_lightgbm(reg, None, init, target_opset=TARGET_OPSET)
        self.assertNotIn('op_type: "Sum"', str(onx))
        oinf = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got1 = oinf.run(None, {"X": X_test})[0]
        assert_almost_equal(expected, got1.ravel(), decimal=5)

        # float split
        onx = convert_lightgbm(reg, None, init, split=10, target_opset=TARGET_OPSET)
        self.assertIn('op_type: "Sum"', str(onx))
        oinf = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got2 = oinf.run(None, {"X": X_test})[0]
        assert_almost_equal(expected, got2.ravel(), decimal=5)

        # final
        d1 = numpy.abs(expected.ravel() - got1.ravel()).mean()
        d2 = numpy.abs(expected.ravel() - got2.ravel()).mean()
        self.assertGreater(d1, d2)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.7.0"),
        reason="Sum<double> not implemented.",
    )
    def test_lightgbm_booster_regressor(self):
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        data = lightgbm.Dataset(X_train, label=y_train)
        model = lightgbm.train(
            {
                "boosting_type": "gbdt",
                "objective": "regression",
                "n_estimators": 100,
                "max_depth": 2,
                "num_thread": 1,
            },
            data,
        )
        expected = model.predict(X_test)
        onx = convert_lightgbm(
            model, "", [("X", FloatTensorType([None, 4]))], target_opset=TARGET_OPSET
        )
        onx10 = convert_lightgbm(
            model,
            "",
            [("X", FloatTensorType([None, 4]))],
            split=1,
            target_opset=TARGET_OPSET,
        )

        self.assertNotIn('op_type: "Sum"', str(onx))
        oinf = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got1 = oinf.run(None, {"X": X_test.astype(numpy.float32)})[0]
        assert_almost_equal(expected, got1.ravel(), decimal=5)

        self.assertIn('op_type: "Sum"', str(onx10))
        oinf = InferenceSession(
            onx10.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got2 = oinf.run(None, {"X": X_test.astype(numpy.float32)})[0]
        assert_almost_equal(expected, got2.ravel(), decimal=5)

        d1 = numpy.abs(expected.ravel() - got1.ravel()).mean()
        d2 = numpy.abs(expected.ravel() - got2.ravel()).mean()
        self.assertGreater(d1, d2)


if __name__ == "__main__":
    unittest.main()
