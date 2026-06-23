# SPDX-License-Identifier: Apache-2.0

import unittest


class TestXGBoostIssuesBig(unittest.TestCase):
    def test_issue_early_stop(self):
        import os
        import pickle
        import onnxruntime
        import numpy as np
        from numpy.testing import assert_almost_equal
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )
        from xgboost import XGBClassifier

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        filename = "test_issue_early_stop.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
        else:
            X, y = make_classification(100000, n_features=20, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

            print(f"training a model X_train.shape={X_train.shape}, X_test={X_test.shape}")

            model = XGBClassifier(
                n_estimators=7500, max_depth=10, early_stopping_rounds=250
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                verbose=1,
            )

            data = dict(
                X_train=X_train,
                X_test=X_test,
                X_val=X_val,
                y_train=y_train,
                y_test=y_test,
                y_val=y_val,
                model=model,
            )
            with open(filename, "wb") as f:
                pickle.dump(data, f)

        # Define input type (adjust shape according to your input)
        X_test, model = data["X_test"], data["model"]
        X_test = X_test[:10]
        initial_type = [("float_input", FloatTensorType([None, X_test.shape[1]]))]
        proba = model.predict_proba(X_test)
        print(proba)

        # Convert XGBoost model to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset={"": 18, "ai.onnx.ml": 3},
            options={"zipmap": False},
        )

        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"float_input": X_test[:10].astype(np.float32)})
        onnx_proba = got[1]
        print(onnx_proba)
        assert_almost_equal(proba, onnx_proba)


if __name__ == "__main__":
    unittest.main(verbosity=2)
