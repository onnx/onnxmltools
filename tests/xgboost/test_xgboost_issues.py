# SPDX-License-Identifier: Apache-2.0
import unittest

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
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 2)

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
    def test_issue_726_binary_logistic_subsample(self):
        import numpy as np
        import onnxruntime as rt
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )

        # overwrite_existing was removed in skl2onnx >=1.18; the default
        # behaviour is already to overwrite, so simply drop the kwarg.
        update_registered_converter(
            XGBRegressor,
            "XGBoostXGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
        )

        X = np.array(
            [[1.0], [2.0], [3.0], [4.0], [2.0], [3.0], [1.0], [2.0]],
            dtype=np.float32,
        )
        y = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)

        model = XGBRegressor(
            max_depth=1,
            n_estimators=3,
            subsample=0.95,
            objective="binary:logistic",
            random_state=0,
        )
        model.fit(X, y)

        initial_types = [("f1", FloatTensorType([None, 1]))]

        onnx_model = convert_sklearn(
            model,
            "XGBoostXGBRegressor",
            initial_types,
            target_opset={"": 13, "ai.onnx.ml": 3},
        )

        sess = rt.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )

        got = sess.run(None, {"f1": X})[0]
        expected = model.predict(X).reshape(-1, 1).astype(np.float32)

        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-5,
            atol=1e-8,
            err_msg=(
                f"\nExpected: {expected.flatten()}"
                f"\nONNX:     {got.flatten()}"
            ),
        )


if __name__ == "__main__":
    unittest.main()