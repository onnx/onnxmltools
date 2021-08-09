import unittest

import numpy as np
from onnx import ModelProto
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm
from onnxruntime import InferenceSession

from lightgbm import LGBMRegressor

_N_DECIMALS=5
_FRAC=0.9999

_y = np.array([0, 0, 1, 1, 1])
_X_train = np.array([[1.0, 0.0], [1.0, -1.0], [1.0, -1.0], [2.0, -1.0], [2.0, -1.0]], dtype=np.float32)
_X_test = np.array([[1.0, np.nan]], dtype=np.float32)

_INITIAL_TYPES = [("input", FloatTensorType([None, _X_train.shape[1]]))]


class TestMissingValues(unittest.TestCase):

    @staticmethod
    def _predict_with_onnx(model: ModelProto, X: np.array) -> np.array:
        session = InferenceSession(model.SerializeToString())
        output_names = [s_output.name for s_output in session.get_outputs()]
        input_names = [s_input.name for s_input in session.get_inputs()]
        if len(input_names) > 1:
            raise RuntimeError(f"Test expects one input. Found multiple inputs: {input_names}.")
        input_name = input_names[0]
        return session.run(output_names, {input_name: X})[0][:, 0]

    @staticmethod
    def _assert_almost_equal(actual: np.array, desired: np.array, decimal: int=7, frac: float=1.0):
        """
        Assert that almost all rows in actual and desired are almost equal to each other.
        Similar to np.testing.assert_almost_equal but allows to define a fraction of rows to be almost
        equal instead of expecting all rows to be almost equal.
        """
        assert 0 <= frac <= 1, "frac must be in range(0, 1)."
        success_abs = (abs(actual - desired) <= (10 ** -decimal)).sum()
        success_rel = success_abs / len(actual)
        assert success_rel >= frac, f"Only {success_abs} out of {len(actual)} rows are almost equal to {decimal} decimals."


    def test_missing_values(self):
        """
        Test that an ONNX model for a LGBM regressor that was trained without having seen missing values
        correctly predicts rows that contain missing values.
        """
        regressor = LGBMRegressor(
            objective="regression",
            min_data_in_bin=1,
            min_data_in_leaf=1,
            n_estimators=1,
            learning_rate=1,
        )
        regressor.fit(_X_train, _y)
        regressor_onnx: ModelProto = convert_lightgbm(regressor, initial_types=_INITIAL_TYPES)
        y_pred = regressor.predict(_X_test)
        y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X_test)
        self._assert_almost_equal(
            y_pred,
            y_pred_onnx,
            decimal=_N_DECIMALS,
            frac=_FRAC,
        )
