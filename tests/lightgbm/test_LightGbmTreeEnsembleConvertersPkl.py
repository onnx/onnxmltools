# SPDX-License-Identifier: Apache-2.0

import sys
import packaging.version as pv
import unittest
import numpy
import pickle
import os
import onnxruntime
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.common.utils import hummingbird_installed
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())

ort_version = ".".join(onnxruntime.__version__.split(".")[:2])


class TestLightGbmTreeEnsembleModelsPkl(unittest.TestCase):
    @unittest.skipIf(
        sys.platform.startswith("win"),
        reason="pickled on linux, may not work on windows",
    )
    @unittest.skipIf(
        sys.platform.startswith("lin"),
        reason="recover linux CI build, needs to be fixed",
    )
    def test_root_leave(self):
        this = os.path.abspath(os.path.dirname(__file__))
        for name in ["example.pkl"]:
            with open(os.path.join(this, name), "rb") as f:
                model = pickle.load(f)
            X = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
            X = numpy.array(X, dtype=numpy.float32)
            model_onnx = convert_lightgbm(
                model.steps[1][1], "pkl1", [("input", FloatTensorType([1, X.shape[1]]))]
            )
            dump_data_and_model(
                X, model.steps[1][1], model_onnx, basename="LightGbmPkl1"
            )

    @unittest.skipIf(
        sys.platform.startswith("win"),
        reason="pickled on linux, may not work on windows",
    )
    @unittest.skipIf(
        sys.platform.startswith("lin"),
        reason="recover linux CI build, needs to be fixed",
    )
    @unittest.skipIf(not hummingbird_installed(), reason="Hummingbird is not installed")
    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.0.0"),
        reason="Hummingbird supports only latest versions of ORT",
    )
    def test_root_leave_onnx_only(self):
        this = os.path.abspath(os.path.dirname(__file__))
        for name in ["example.pkl"]:
            with open(os.path.join(this, name), "rb") as f:
                model = pickle.load(f)
            X = [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
            X = numpy.array(X, dtype=numpy.float32)
            model_onnx = convert_lightgbm(
                model.steps[1][1],
                "pkl1",
                [("input", FloatTensorType([1, X.shape[1]]))],
                without_onnx_ml=True,
                target_opset=TARGET_OPSET,
            )
            dump_data_and_model(
                X, model.steps[1][1], model_onnx, basename="LightGbmPkl1"
            )


if __name__ == "__main__":
    unittest.main()
