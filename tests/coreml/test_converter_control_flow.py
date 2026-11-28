# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from onnxmltools.convert.coreml.operator_converters.SVC import _get_kernel_params
from onnxmltools.convert.coreml.operator_converters.neural_network.SequenceRepeat import (
    convert_sequence_repeat,
)


class TestCoreMLConverterControlFlow(unittest.TestCase):
    def test_supported_svm_kernel_parameters(self):
        kernel = SimpleNamespace(
            rbfKernel=SimpleNamespace(gamma=0.25),
            polyKernel=SimpleNamespace(gamma=0.5, coef0=1.5, degree=3),
            sigmoidKernel=SimpleNamespace(gamma=0.75, coef0=2.5),
        )
        expected = {
            "linearKernel": [0.0, 0.0, 0.0],
            "rbfKernel": [0.25, 0.0, 0.0],
            "polyKernel": [0.5, 1.5, 3],
            "sigmoidKernel": [0.75, 2.5, 0.0],
        }

        for kernel_type, kernel_params in expected.items():
            with self.subTest(kernel_type=kernel_type):
                self.assertEqual(_get_kernel_params(kernel, kernel_type), kernel_params)

    def test_precomputed_svm_kernel_raises(self):
        with self.assertRaisesRegex(ValueError, "not supported by ONNX-ML"):
            _get_kernel_params(SimpleNamespace(), "precomputedKernel")

    def test_unsupported_svm_kernel_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported SVM kernel type"):
            _get_kernel_params(SimpleNamespace(), "unknownKernel")

    def test_sequence_repeat_supported_ranks(self):
        for shape, expected_repeats in (
            ([None, 4], [3, 1]),
            ([None, 3, 5, 5], [3, 1, 1, 1]),
        ):
            with self.subTest(shape=shape):
                operator = self._make_sequence_repeat_operator(shape)
                scope = object()
                container = object()
                with patch(
                    "onnxmltools.convert.coreml.operator_converters.neural_network."
                    "SequenceRepeat.apply_tile"
                ) as apply_tile:
                    convert_sequence_repeat(scope, operator, container)

                apply_tile.assert_called_once_with(
                    scope,
                    "input",
                    "output",
                    container,
                    operator_name="repeat",
                    repeats=expected_repeats,
                )

    def test_sequence_repeat_rejects_unsupported_rank(self):
        operator = self._make_sequence_repeat_operator([None, 3, 5])

        with self.assertRaisesRegex(RuntimeError, "must be either 2-D or 4-D"):
            convert_sequence_repeat(object(), operator, object())

    @staticmethod
    def _make_sequence_repeat_operator(shape):
        return SimpleNamespace(
            raw_operator=SimpleNamespace(
                sequenceRepeat=SimpleNamespace(nRepetitions=3)
            ),
            inputs=[SimpleNamespace(type=SimpleNamespace(shape=shape))],
            input_full_names=["input"],
            output_full_names=["output"],
            full_name="repeat",
        )


if __name__ == "__main__":
    unittest.main()
