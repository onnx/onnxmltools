# SPDX-License-Identifier: Apache-2.0

"""
Tests scilit-learn's tree-based methods' converters.
"""
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import pandas
import onnxruntime as rt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxconverter_common import data_types as onnxtypes
from onnxmltools.convert import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost as convert_xgb,
)

try:
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import (
        calculate_linear_regressor_output_shapes,
    )

    can_test = True
except ImportError:
    # sklearn-onnx not recent enough
    can_test = False


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
TARGET_OPSET_ML = 2


@unittest.skipIf(sys.version_info[:2] <= (3, 5), reason="not available")
@unittest.skipIf(not can_test, reason="sklearn-onnx not recent enough")
class TestXGBoostModelsPipeline(unittest.TestCase):
    def _column_tranformer_fitted_from_df(self, data):
        def transformer_for_column(column):
            if column.dtype in ["float64", "float32"]:
                return MinMaxScaler()
            if column.dtype in ["bool"]:
                return "passthrough"
            if column.dtype in ["O"]:
                return OneHotEncoder(sparse_output=False)
            raise ValueError()

        return ColumnTransformer(
            [(col, transformer_for_column(data[col]), [col]) for col in data.columns],
            remainder="drop",
        ).fit(data)

    def _convert_dataframe_schema(self, data):
        def type_for_column(column):
            if column.dtype in ["float64", "float32"]:
                return onnxtypes.FloatTensorType([None, 1])
            if column.dtype in ["int64"]:
                return onnxtypes.Int64TensorType([None, 1])
            if column.dtype in ["bool"]:
                return onnxtypes.BooleanTensorType([None, 1])
            if column.dtype in ["O"]:
                return onnxtypes.StringTensorType([None, 1])
            raise ValueError()

        res = [(col, type_for_column(data[col])) for col in data.columns]
        return res

    def test_xgboost_10_skl_missing(self):
        self.common_test_xgboost_10_skl(np.nan)

    def test_xgboost_10_skl_zero(self):
        try:
            self.common_test_xgboost_10_skl(0.0, True)
        except RuntimeError as e:
            assert "Cannot convert a XGBoost model where missing values" in str(e)

    def test_xgboost_10_skl_zero_replace(self):
        self.common_test_xgboost_10_skl(np.nan, True)

    def common_test_xgboost_10_skl(self, missing, replace=False):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data_fail.csv")
        data = pandas.read_csv(data)

        for col in data:
            dtype = data[col].dtype
            if dtype in ["float64", "float32"]:
                data[col].fillna(0.0, inplace=True)
            if dtype in ["int64"]:
                data[col].fillna(0, inplace=True)
            elif dtype in ["O"]:
                data[col].fillna("N/A", inplace=True)

        data["pclass"] = data["pclass"] * float(1)
        full_df = data.drop("survived", axis=1)
        full_labels = data["survived"]

        train_df, test_df, train_labels, test_labels = train_test_split(
            full_df, full_labels, test_size=0.2, random_state=11
        )

        col_transformer = self._column_tranformer_fitted_from_df(full_df)

        param_distributions = {
            "colsample_bytree": 0.5,
            "gamma": 0.2,
            "learning_rate": 0.3,
            "max_depth": 2,
            "min_child_weight": 1.0,
            "n_estimators": 1,
            "missing": missing,
        }

        regressor = XGBRegressor(
            verbose=0, objective="reg:squarederror", **param_distributions
        )
        regressor.fit(col_transformer.transform(train_df), train_labels)
        model = Pipeline(
            steps=[("preprocessor", col_transformer), ("regressor", regressor)]
        )

        update_registered_converter(
            XGBRegressor,
            "XGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgb,
        )

        # last step
        input_xgb = model.steps[0][-1].transform(test_df[:5]).astype(np.float32)
        if replace:
            input_xgb[input_xgb[:, :] == missing] = np.nan
        onnx_last = convert_sklearn(
            model.steps[1][-1],
            initial_types=[("X", FloatTensorType(shape=[None, input_xgb.shape[1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        session = rt.InferenceSession(
            onnx_last.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        pred_skl = model.steps[1][-1].predict(input_xgb).ravel()
        pred_onx = session.run(None, {"X": input_xgb})[0].ravel()
        assert_almost_equal(pred_skl, pred_onx)


if __name__ == "__main__":
    unittest.main()
