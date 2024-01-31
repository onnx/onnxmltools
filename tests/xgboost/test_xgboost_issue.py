# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime import InferenceSession


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestXGBoostIssue(unittest.TestCase):
    def common_test(self, cls, n_estimators):
        dataset = load_iris()
        X, y = dataset.data, dataset.target
        model = cls(
            n_estimators=n_estimators,
            learning_rate=1.0,
            subsample=0.8,
            colsample_bynode=0.8,
            reg_lambda=1e-5,
        )
        model.fit(X, y)
        data = np.random.rand(5, 4).astype(np.float32)
        expected_labels = model.predict(data)
        expected_probabilities = (
            model.predict_proba(data) if hasattr(model, "predict_proba") else None
        )

        onnx_model = convert_xgboost(
            model, initial_types=[("input", FloatTensorType(shape=[None, None]))]
        )

        session = InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        if expected_probabilities is None:
            (onnx_predictions,) = session.run(None, {"input": data})
            assert_almost_equal(expected_labels, onnx_predictions.ravel())
        else:
            onnx_predictions, onnx_probabilities = session.run(None, {"input": data})
            assert_almost_equal(expected_probabilities, onnx_probabilities)
            assert_almost_equal(expected_labels, onnx_predictions.ravel())

    def test_issue_663_classifier(self):
        self.common_test(XGBClassifier, 1)
        self.common_test(XGBRFClassifier, 1)
        self.common_test(XGBClassifier, 2)
        self.common_test(XGBRFClassifier, 2)

    def test_issue_663_regressor(self):
        self.common_test(XGBRegressor, 1)
        self.common_test(XGBRFRegressor, 1)
        self.common_test(XGBRegressor, 2)
        self.common_test(XGBRFRegressor, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
