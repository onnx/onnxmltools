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
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestLightGbmTreeEnsembleModelsPkl(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2, reason="pickled with Python 3, cannot unpickle with 2")
    @unittest.skipIf(sys.platform.startswith('win'), reason="pickled on linux, may not work on windows")
    def test_root_leave(self):
        this = os.path.abspath(os.path.dirname(__file__))
        for name in ["example.pkl"]:
            with open(os.path.join(this, name), "rb") as f:
                model = pickle.load(f)
            X = [[0., 1.], [1., 1.], [2., 0.]]
            X = numpy.array(X, dtype=numpy.float32)
            model_onnx = convert_lightgbm(model.steps[1][1], 'pkl1', [('input', FloatTensorType([1, X.shape[1]]))])
            dump_data_and_model(X, model.steps[1][1], model_onnx, basename="LightGbmPkl1")


if __name__ == "__main__":
    unittest.main()
