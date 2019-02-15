# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import sys
import unittest
import numpy
import pickle
import os
from lightgbm import LGBMClassifier
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestLightGbmTreeEnsembleModelsPkl(unittest.TestCase):

    def test_root_leave(self):
        this = os.path.abspath(os.path.dirname(__file__))
        
        model = LGBMClassifier(boosting_type='gbdt', class_weight=None,
                    colsample_bytree=0.6933333333333332, importance_type='split',
                    learning_rate=0.036848421052631586, max_bin=210, max_depth=8,
                    min_child_samples=3, min_child_weight=9,
                    min_split_gain=0.631578947368421, n_estimators=800, n_jobs=1,
                    num_leaves=173, objective=None, random_state=None,
                    reg_alpha=0.631578947368421, reg_lambda=0.5263157894736842,
                    silent=True, subsample=0.9405263157894738,
                    subsample_for_bin=200000, subsample_freq=0, verbose=-10)
        
        X = numpy.array([[0., 1.], [1., 1.], [2., 0.]])
        Y = numpy.array([0, 1, 0])
        model.fit(X, Y)
        X = numpy.array(X, dtype=numpy.float32)
        model_onnx = convert_lightgbm(model, 'pkl1', [('input', FloatTensorType([1, X.shape[1]]))])
        dump_data_and_model(X, model, model_onnx, basename="LightGbmPkl1")


if __name__ == "__main__":
    unittest.main()
