# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from onnxmltools.utils import dump_one_class_classification, dump_binary_classification, dump_multiple_classification
from onnxmltools.utils import dump_multiple_regression, dump_single_regression


class TestSklearnDecisionTreeModels(unittest.TestCase):

    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        dump_one_class_classification(model)
        dump_binary_classification(model)
        dump_multiple_classification(model)

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        dump_single_regression(model)
        dump_multiple_regression(model)


if __name__ == "__main__":
    unittest.main()
