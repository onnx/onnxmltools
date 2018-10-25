# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import numpy
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
from onnxmltools.utils import dump_data_and_model


class TestLightGbmTreeEnsembleModels(unittest.TestCase):
    def _test_one_class_classification_core(self, model, opts=""):
        X = [[0., 1.], [1., 1.], [2., 0.]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [1, 1, 1]
        model.fit(X, y)
        model_onnx = convert_lightgbm(model, 'tree-based classifier', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="LightGbmOne" + model.__class__.__name__ + opts)

    def _test_binary_classification_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = ['A', 'B', 'A']
        model.fit(X, y)
        model_onnx = convert_lightgbm(model, 'tree-based binary classifier', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="LightGbmBin" + model.__class__.__name__ + opts)

    def _test_multiple_classification_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        X = numpy.array(X, dtype=numpy.float32)
        y = [0, 1, 2, 1, 1, 2]
        model.fit(X, y)
        model_onnx = convert_lightgbm(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="LightGbmMcl" + model.__class__.__name__ + opts)

    def _test_multiple_regression_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([[100, 50], [100, 49], [100, 99]], dtype=numpy.float32)
        model.fit(X, y)
        model_onnx = convert_lightgbm(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="LightGbmMRg" + model.__class__.__name__ + opts)

    def _test_single_regression_core(self, model, opts=""):
        X = [[0, 1], [1, 1], [2, 0]]
        X = numpy.array(X, dtype=numpy.float32)
        y = numpy.array([100, -10, 50], dtype=numpy.float32)
        model.fit(X, y)
        model_onnx = convert_lightgbm(model, 'tree-based regressor', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X, model, model_onnx, basename="LightGbmReg" + model.__class__.__name__ + opts)

    def test_lightgbm_classifier(self):
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        self._test_binary_classification_core(model)
        self._test_multiple_classification_core(model)

    def test_lightgbm_regressor(self):
        model = LGBMRegressor(n_estimators=3, min_child_samples=1)
        self._test_single_regression_core(model)

    def test_lightgbm_regressor1(self):
        model = LGBMRegressor(n_estimators=1, min_child_samples=1)
        self._test_single_regression_core(model, opts="1")

    def test_lightgbm_regressor2(self):
        model = LGBMRegressor(n_estimators=2, max_depth=1, min_child_samples=1)
        self._test_single_regression_core(model, opts="2")



if __name__ == "__main__":
    unittest.main()
