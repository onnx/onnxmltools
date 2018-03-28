"""
Tests scilit-learn's tree-based methods' converters.
"""
import unittest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from onnxmltools import convert_sklearn
from onnxmltools.convert.coreml._data_types import FloatTensorType, Int64TensorType


class TestSklearnTreeEnsembleModels(unittest.TestCase):
    def _test_one_class_classification_core(self, model):
        X = [[0, 1], [1, 1], [2, 0]]
        y = [1, 1, 1]
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based classifier', [Int64TensorType([1, 2])])
        self.assertTrue(model_onnx is not None)

    def _test_binary_classification_core(self, model):
        X = [[0, 1], [1, 1], [2, 0]]
        y = ['A', 'B', 'A']
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based binary classifier', [Int64TensorType([1, 2])])
        self.assertTrue(model_onnx is not None)

    def _test_multiple_output_core(self, model):
        X = [[0, 1], [1, 1], [2, 0]]
        y = [[100, 50], [100, 49], [100, 99]]
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based multi-output regressor', [Int64TensorType([1, 2])])
        self.assertTrue(model_onnx is not None)

    def _test_single_output_core(self, model):
        X = [[0, 1], [1, 1], [2, 0]]
        y = [100, -10, 50]
        model.fit(X, y)
        model_onnx = convert_sklearn(model, 'tree-based regressor', [Int64TensorType([1, 2])])
        self.assertTrue(model_onnx is not None)

    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        self._test_one_class_classification_core(model)
        self._test_binary_classification_core(model)
        self._test_single_output_core(model)
        self._test_multiple_output_core(model)

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        self._test_single_output_core(model)
        self._test_multiple_output_core(model)

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        self._test_one_class_classification_core(model)
        self._test_binary_classification_core(model)
        self._test_single_output_core(model)
        self._test_multiple_output_core(model)

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        self._test_single_output_core(model)
        self._test_multiple_output_core(model)

    def test_gradient_boosting_classifier(self):
        model = GradientBoostingClassifier(n_estimators=3)
        self._test_binary_classification_core(model)
        self._test_single_output_core(model)

    def test_gradient_boosting_regressor(self):
        model = GradientBoostingRegressor(n_estimators=3)
        self._test_single_output_core(model)
