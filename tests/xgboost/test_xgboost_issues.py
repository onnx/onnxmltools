# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from numpy.testing import assert_allclose

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class TestXGBoostIssues(unittest.TestCase):
    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_issue_676(self):
        import json
        import onnxruntime
        import xgboost
        import numpy as np
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import update_registered_converter
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )

        def xgbregressor_shape_calculator(operator):
            config = json.loads(operator.raw_operator.get_booster().save_config())
            n_targets = int(config["learner"]["learner_model_param"]["num_target"])
            operator.outputs[0].type.shape = [None, n_targets]

        update_registered_converter(
            xgboost.XGBRegressor,
            "XGBoostXGBRegressor",
            xgbregressor_shape_calculator,
            convert_xgboost,
        )
        # Your data and labels
        rng = np.random.default_rng()
        X = rng.random((100, 10))
        y = rng.random((100, 2))

        # Train XGBoost regressor
        model = xgboost.XGBRegressor(
            objective="reg:squarederror", n_estimators=2, maxdepth=2
        )
        model.fit(X, y)

        # Define input type (adjust shape according to your input)
        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]

        # Convert XGBoost model to ONNX
        onnx_model = convert_sklearn(
            model, initial_types=initial_type, target_opset={"": 12, "ai.onnx.ml": 3}
        )
        self.assertIn("dim_value: 2", str(onnx_model.graph.output))

        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"float_input": X.astype(np.float32)})
        self.assertEqual(got[0].shape, (100, 2))

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_issue_676_values(self):
        import onnxruntime
        import xgboost
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        rng = np.random.default_rng(0)
        X = rng.random((50, 10)).astype(np.float32)
        y = rng.random((50, 10))

        model = xgboost.XGBRegressor(objective="reg:squarederror", n_estimators=3)
        model.fit(X, y)

        onnx_model = convert_xgboost(
            model, initial_types=[("float_input", FloatTensorType([None, 10]))]
        )
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"float_input": X})[0]
        expected = model.predict(X)

        self.assertEqual(got.shape, (50, 10))
        assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_quantile_regression(self):
        import onnxruntime
        import xgboost
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        rng = np.random.default_rng(0)
        X = rng.random((20, 3)).astype(np.float32)
        y = rng.random(20)

        model = xgboost.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=[0.1, 0.5, 0.9],
            n_estimators=3,
        )
        model.fit(X, y)

        onnx_model = convert_xgboost(
            model, initial_types=[("input", FloatTensorType([None, 3]))]
        )
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": X})[0]
        expected = model.predict(X)

        self.assertEqual(got.shape, (20, 3))
        assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
