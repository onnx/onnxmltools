# SPDX-License-Identifier: Apache-2.0

import os
import packaging.version as pv
import unittest
import pickle
import xgboost
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert.xgboost import convert as convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestXGBoostUnpickle06(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(xgboost.__version__) >= pv.Version("1.0"),
        reason="compatibility break with pickle in 1.0",
    )
    def test_xgboost_unpickle_06(self):
        # Unpickle a model trained with an old version of xgboost.
        this = os.path.dirname(__file__)
        with open(os.path.join(this, "xgboost10day.pickle.dat"), "rb") as f:
            xgb = pickle.load(f)

        conv_model = convert_xgboost(
            xgb,
            initial_types=[("features", FloatTensorType(["None", 10000]))],
            target_opset=TARGET_OPSET,
        )
        assert conv_model is not None


if __name__ == "__main__":
    unittest.main()
