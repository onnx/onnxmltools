# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from onnxmltools.utils import dump_binary_classification
from onnxmltools.utils import dump_single_regression


class TestSklearnGradientBoostingModels(unittest.TestCase):

    def test_gradient_boosting_classifier(self):
        model = GradientBoostingClassifier(n_estimators=3)
        dump_binary_classification(model)

    def test_gradient_boosting_regressor(self):
        model = GradientBoostingRegressor(n_estimators=3)
        dump_single_regression(model)


if __name__ == "__main__":
    unittest.main()
