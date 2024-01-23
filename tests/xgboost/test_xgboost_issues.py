# SPDX-License-Identifier: Apache-2.0

import unittest


class TestXGBoostIssues(unittest.TestCase):
    def test_issue_676(self):
        import onnxruntime
        import xgboost
        import numpy as np
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import update_registered_converter
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )

        def frozen_shape_calculator(operator):
            operator.outputs[0].type.shape = [None, 2]

        update_registered_converter(
            xgboost.XGBRegressor,
            "XGBoostXGBRegressor",
            frozen_shape_calculator,
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
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        self.assertIn("dim_value: 2", str(onnx_model.graph.output))

        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"float_input": X.astype(np.float32)})
        self.assertEqual(got[0].shape, (100, 2))


if __name__ == "__main__":
    unittest.main()
