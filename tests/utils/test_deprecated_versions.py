# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings

import numpy as np
import packaging.version as pv

from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.common.utils import lightgbm_installed, xgboost_installed
from onnxmltools.convert.main import _warn_if_deprecated_version


def _future_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func(*args, **kwargs)
    return [
        w
        for w in caught
        if issubclass(w.category, FutureWarning) and "onnxmltools 1.18.0" in str(w.message)
    ]


class TestDeprecatedVersions(unittest.TestCase):
    def test_warns_below_minimum(self):
        module = types.SimpleNamespace(__name__="xgboost", __version__="1.7.6")
        caught = _future_warnings(_warn_if_deprecated_version, module, "2.0.0")
        self.assertEqual(len(caught), 1)
        self.assertIn("xgboost 1.7.6 is deprecated", str(caught[0].message))
        self.assertIn("onnxmltools 1.18.0", str(caught[0].message))

    def test_silent_at_or_above_minimum(self):
        for version in ("2.0.0", "3.4.1"):
            module = types.SimpleNamespace(__name__="xgboost", __version__=version)
            self.assertEqual(
                _future_warnings(_warn_if_deprecated_version, module, "2.0.0"), []
            )

    def test_silent_without_version(self):
        module = types.SimpleNamespace(__name__="xgboost")
        self.assertEqual(_future_warnings(_warn_if_deprecated_version, module, "2.0.0"), [])

    @unittest.skipIf(not lightgbm_installed(), reason="lightgbm not installed")
    def test_convert_lightgbm(self):
        import lightgbm
        from onnxmltools import convert_lightgbm

        X = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        model = lightgbm.LGBMRegressor(n_estimators=2, verbose=-1).fit(X, X[:, 0])
        caught = _future_warnings(
            convert_lightgbm, model, initial_types=[("X", FloatTensorType([None, 3]))]
        )
        expected = 1 if pv.Version(lightgbm.__version__) < pv.Version("4.0.0") else 0
        self.assertEqual(len(caught), expected)

    @unittest.skipIf(not xgboost_installed(), reason="xgboost not installed")
    def test_convert_xgboost(self):
        import xgboost
        from onnxmltools import convert_xgboost

        X = np.random.RandomState(0).rand(20, 3).astype(np.float32)
        model = xgboost.XGBRegressor(n_estimators=2).fit(X, X[:, 0])
        caught = _future_warnings(
            convert_xgboost, model, initial_types=[("X", FloatTensorType([None, 3]))]
        )
        expected = 1 if pv.Version(xgboost.__version__) < pv.Version("2.0.0") else 0
        self.assertEqual(len(caught), expected)


if __name__ == "__main__":
    unittest.main()
