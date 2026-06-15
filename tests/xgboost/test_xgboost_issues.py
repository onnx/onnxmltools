# SPDX-License-Identifier: Apache-2.0

import unittest

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class TestXGBoostIssues(unittest.TestCase):
    @unittest.skipIf(XGBRegressor is None, "xgboost is not available")
    def test_xgbregressor_binary_logistic_with_subsample(self):
        """
        XGBRegressor with binary:logistic and subsample<1.0 should produce
        results matching XGBoost's predict() output. The base_score in the
        model config is stored in probability space, but leaf values are in
        log-odds space; the converter must apply logit(base_score) and then
        a Sigmoid node to replicate XGBoost's sigmoid(logit(base) + leaves).
        """
        import numpy as np
        import pandas as pd
        import onnxruntime
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 2.0],
                "label": [1, 0, 1, 0, 1, 1, 0, 1],
            }
        )
        X_df = df.drop(columns=["label"])
        y = df["label"]
        X_np = df["f1"].values.reshape(-1, 1).astype(np.float32)

        for subsample in [1.0, 0.95]:
            params = {
                "max_depth": 1,
                "n_estimators": 3,
                "subsample": subsample,
                "objective": "binary:logistic",
                "random_state": 42,
            }
            model = XGBRegressor(**params)
            model.fit(X_df, y)

            initial_types = [("f1", FloatTensorType([None, 1]))]
            onnx_model = convert_xgboost(
                model,
                "XGBoostXGBRegressor",
                initial_types,
                target_opset=13,
            )

            sess = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
            onnx_output = sess.run(None, {"f1": X_np})[0]
            expected = model.predict(X_df).reshape(-1, 1).astype(np.float32)

            np.testing.assert_allclose(
                onnx_output,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ONNX output mismatch for subsample={subsample}",
            )

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


if __name__ == "__main__":
    unittest.main()
