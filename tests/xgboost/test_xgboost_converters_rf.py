# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from sklearn.datasets import load_diabetes, make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor, XGBRFClassifier
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def fct_cl2(y):
    y[y == 2] = 0
    return y


def fct_cl3(y):
    y[y == 0] = 6
    return y


def fct_id(y):
    return y


def _fit_classification_model(model, n_classes, is_str=False, dtype=None):
    x, y = make_classification(
        n_classes=n_classes,
        n_features=100,
        n_samples=1000,
        random_state=42,
        n_informative=7,
    )
    y = y.astype(np.str_) if is_str else y.astype(np.int64)
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.5, random_state=42)
    if dtype is not None:
        y_train = y_train.astype(dtype)
    model.fit(x_train, y_train)
    return model, x_test.astype(np.float32)


class TestXGBoostRFModels(unittest.TestCase):
    def test_xgbrf_aregressor(self):
        iris = load_diabetes()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.5, random_state=42
        )
        xgb = XGBRFRegressor()
        xgb.fit(x_train, y_train)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            x_test.astype("float32"),
            xgb,
            conv_model,
            basename="SklearnXGBRFRegressor-Dec3",
        )

    def test_xgbrf_classifier(self):
        xgb, x_test = _fit_classification_model(XGBRFClassifier(), 2)
        conv_model = convert_xgboost(
            xgb,
            initial_types=[("input", FloatTensorType(shape=[None, None]))],
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(x_test, xgb, conv_model, basename="SklearnXGBRFClassifier")


if __name__ == "__main__":
    # TestXGBoostModels().test_xgboost_booster_classifier_multiclass_softprob()
    unittest.main(verbosity=2)
