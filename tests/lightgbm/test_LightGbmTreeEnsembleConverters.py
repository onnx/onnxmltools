# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model
from onnxmltools.utils import dump_one_class_classification, dump_binary_classification, dump_multiple_classification
from onnxmltools.utils import dump_multiple_regression, dump_single_regression


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


if __name__ == "__main__":
    unittest.main()
