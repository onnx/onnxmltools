# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from onnxmltools.utils.utils_backend import OnnxRuntimeAssertionError
from onnxmltools.utils.utils_backend_onnxruntime import compare_runtime


class TestOnnxRuntimeBackend(unittest.TestCase):
    @patch("onnxruntime.InferenceSession")
    def test_one_off_runtime_error_identifies_model(self, inference_session):
        session = MagicMock()
        session.get_inputs.return_value = [SimpleNamespace(name="input")]
        session.run.side_effect = RuntimeError("test failure")
        inference_session.return_value = session

        test = {
            "onnx": "test-model.onnx",
            "data": [np.array([[1.0]], dtype=np.float32)],
            "expected": [],
        }

        with self.assertRaisesRegex(
            OnnxRuntimeAssertionError, "test-model.onnx.*test failure"
        ):
            compare_runtime(test, options={"OneOff": True})

    @patch("onnxruntime.InferenceSession")
    def test_input_shape_error_identifies_model(self, inference_session):
        session = MagicMock()
        session.get_inputs.return_value = [
            SimpleNamespace(name="first", shape=[None, 2]),
            SimpleNamespace(name="second", shape=[None, 3]),
        ]
        inference_session.return_value = session

        test = {
            "onnx": "shape-model.onnx",
            "data": np.zeros((1, 4), dtype=np.float32),
            "expected": [],
        }

        with self.assertRaisesRegex(OnnxRuntimeAssertionError, "shape-model.onnx"):
            compare_runtime(test, options={})


if __name__ == "__main__":
    unittest.main()
