# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import lightgbm
import numpy
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from onnxmltools.utils import dump_binary_classification, dump_multiple_classification
from onnxmltools.utils import dump_single_regression
from onnxmltools.utils.tests_helper import convert_model


class TestLightGbmTreeEnsembleModels(unittest.TestCase):

    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        dump_binary_classification(model, allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")
        dump_multiple_classification(model, allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        dump_single_regression(model)

    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        dump_single_regression(model, suffix="1")

    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        dump_single_regression(model, suffix="2")

    def test_lightgbm_booster_classifier(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 0, 1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'binary',
                                'n_estimators': 3, 'min_child_samples': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based multi-output classifier',
                                           [('input', FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')",
                            basename=prefix + "BoosterBin" + model.__class__.__name__)

    def test_lightgbm_booster_regressor(self):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 1.1]
        data = lightgbm.Dataset(X, label=y)
        model = lightgbm.train({'boosting_type': 'gbdt', 'objective': 'regression',
                                'n_estimators': 3, 'min_child_samples': 1, 'max_depth': 1},
                               data)
        model_onnx, prefix = convert_model(model, 'tree-based binary classifier',
                                           [('input', FloatTensorType([None, 2]))])
        dump_data_and_model(X, model, model_onnx,
                            basename=prefix + "BoosterBin" + model.__class__.__name__)


if __name__ == "__main__":
    unittest.main()
