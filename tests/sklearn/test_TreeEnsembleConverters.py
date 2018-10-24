# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnTreeEnsembleModels(unittest.TestCase):
    def _test_one_class_classification_core(self, model, opts=""):
        X = [[0., 1.], [1., 1.], [2., 0.]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [1, 1, 1]
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based classifier', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="SklearnOne" + model.__class__.__name__ + opts)

    def _test_binary_classification_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = ['A', 'B', 'A']
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based binary classifier', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="SklearnBin" + model.__class__.__name__ + opts)

    def _test_multiple_classification_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMcl" + model.__class__.__name__ + opts)

    def _test_multiple_regression_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([[100, 50], [100, 49], [100, 99]], dtype=numpy.float32)
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMRg" + model.__class__.__name__ + opts)

    def _test_single_regression_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="SklearnReg" + model.__class__.__name__ + opts)

    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        self._test_one_class_classification_core(model)
        self._test_binary_classification_core(model)
        self._test_multiple_classification_core(model)

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        self._test_single_regression_core(model)
        self._test_multiple_regression_core(model)

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        self._test_one_class_classification_core(model)
        self._test_binary_classification_core(model)
        self._test_multiple_classification_core(model)

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        self._test_single_regression_core(model)
        self._test_multiple_regression_core(model)

    def test_extra_trees_classifier(self):
        model = ExtraTreesClassifier(n_estimators=3)
        self._test_one_class_classification_core(model)
        self._test_binary_classification_core(model)
        self._test_multiple_classification_core(model)

    def test_extra_trees_regressor(self):
        model = ExtraTreesRegressor(n_estimators=3)
        self._test_single_regression_core(model)
        self._test_multiple_regression_core(model)

    def test_gradient_boosting_classifier(self):
        model = GradientBoostingClassifier(n_estimators=3)
        self._test_binary_classification_core(model)

    def test_gradient_boosting_regressor(self):
        model = GradientBoostingRegressor(n_estimators=3)
        self._test_single_regression_core(model)

    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        self._test_binary_classification_core(model)
        self._test_multiple_classification_core(model)

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        self._test_single_regression_core(model)


if __name__ == "__main__":
    unittest.main()
