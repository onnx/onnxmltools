# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import sys
from distutils.version import StrictVersion
import unittest
import numpy
import pickle
import os
import onnxruntime
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.common.utils import hummingbird_installed
from onnxmltools.utils import dump_data_and_model


class TestLightGbmTreeEnsembleModelsPkl(unittest.TestCase):

    @unittest.skipIf(sys.version_info[0] == 2, reason="pickled with Python 3, cannot unpickle with 2")
    @unittest.skipIf(sys.platform.startswith('win'), reason="pickled on linux, may not work on windows")
    @unittest.skipIf(sys.platform.startswith('lin'), reason="recover linux CI build, needs to be fixed")
    def test_root_leave(self):
        this = os.path.abspath(os.path.dirname(__file__))
        for name in ["example.pkl"]:
            with open(os.path.join(this, name), "rb") as f:
                model = pickle.load(f)
            X = [[0., 1.], [1., 1.], [2., 0.]]
            X = numpy.array(X, dtype=numpy.float32)
            model_onnx = convert_lightgbm(model.steps[1][1], 'pkl1', [('input', FloatTensorType([1, X.shape[1]]))])
            dump_data_and_model(X, model.steps[1][1], model_onnx, basename="LightGbmPkl1")
    

    @unittest.skipIf(sys.version_info[0] == 2, reason="pickled with Python 3, cannot unpickle with 2")
    @unittest.skipIf(sys.platform.startswith('win'), reason="pickled on linux, may not work on windows")
    @unittest.skipIf(sys.platform.startswith('lin'), reason="recover linux CI build, needs to be fixed")
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        StrictVersion(onnxruntime.__version__) < StrictVersion('1.0.0'), reason="Hummingbird supports only latest versions of ORT"
    )
    def test_root_leave_onnx_only(self):
        this = os.path.abspath(os.path.dirname(__file__))
        for name in ["example.pkl"]:
            with open(os.path.join(this, name), "rb") as f:
                model = pickle.load(f)
            X = [[0., 1.], [1., 1.], [2., 0.]]
            X = numpy.array(X, dtype=numpy.float32)
            model_onnx = convert_lightgbm(model.steps[1][1], 'pkl1', [('input', FloatTensorType([1, X.shape[1]]))], without_onnx_ml=True)
            dump_data_and_model(X, model.steps[1][1], model_onnx, basename="LightGbmPkl1")


if __name__ == "__main__":
    unittest.main()
