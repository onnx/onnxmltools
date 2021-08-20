import unittest
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime
import pandas as pd
from onnx import ModelProto
from onnxconverter_common.data_types import DoubleTensorType, TensorType
from onnxmltools import convert_lightgbm
from onnxruntime import InferenceSession
from pandas.core.frame import DataFrame

from lightgbm import LGBMRegressor

_N_ROWS=10_000
_N_COLS=10
_N_DECIMALS=5
_FRAC = 0.9997

_X = pd.DataFrame(np.random.random(size=(_N_ROWS, _N_COLS)))
_Y = pd.Series(np.random.random(size=_N_ROWS))

_DTYPE_MAP: Dict[str, TensorType] = {
    "float64": DoubleTensorType,
}


class ObjectiveTest(unittest.TestCase):

    _objectives: Tuple[str] = (
        "regression",
        "poisson",
        "gamma",
    )

    @staticmethod
    def _calc_initial_types(X: DataFrame) -> List[Tuple[str, TensorType]]:
        dtypes = set(str(dtype) for dtype in X.dtypes)
        if len(dtypes) > 1:
            raise RuntimeError(f"Test expects homogenous input matrix. Found multiple dtypes: {dtypes}.")
        dtype = dtypes.pop()
        tensor_type = _DTYPE_MAP[dtype]
        return [("input", tensor_type(X.shape))]

    @staticmethod
    def _predict_with_onnx(model: ModelProto, X: DataFrame) -> np.array:
        session = InferenceSession(model.SerializeToString())
        output_names = [s_output.name for s_output in session.get_outputs()]
        input_names = [s_input.name for s_input in session.get_inputs()]
        if len(input_names) > 1:
            raise RuntimeError(f"Test expects one input. Found multiple inputs: {input_names}.")
        input_name = input_names[0]
        return session.run(output_names, {input_name: X.values})[0][:, 0]

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

    @unittest.skipIf(tuple(int(ver) for ver in onnxruntime.__version__.split(".")) < (1, 3), "not supported in this library version")
    def test_objective(self):
        """
        Test if a LGBMRegressor a with certain objective (e.g. 'poisson') can be converted to ONNX
        and whether the ONNX graph and the original model produce almost equal predictions.

        Note that this tests is a bit flaky because of precision differences with ONNX and LightGBM
        and therefore sometimes fails randomly. In these cases, a retry should resolve the issue.
        """
        for objective in self._objectives:
            with self.subTest(X=_X, objective=objective):
                regressor = LGBMRegressor(objective=objective)
                regressor.fit(_X, _Y)
                regressor_onnx: ModelProto = convert_lightgbm(regressor, initial_types=self._calc_initial_types(_X))
                y_pred = regressor.predict(_X)
                y_pred_onnx = self._predict_with_onnx(regressor_onnx, _X)
                self._assert_almost_equal(
                    y_pred,
                    y_pred_onnx,
                    decimal=_N_DECIMALS,
                    frac=_FRAC,
                )
