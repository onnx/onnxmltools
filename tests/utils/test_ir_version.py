# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import onnx
from onnx import TensorProto, checker, helper
from onnxruntime import InferenceSession

from onnxmltools.convert.common._topology import make_model_ex


def identity_graph(initializer=False):
    inputs = (
        [] if initializer else [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])]
    )
    initializers = (
        [helper.make_tensor("X", TensorProto.FLOAT, [2], [1.0, 2.0])] if initializer else []
    )
    return helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "identity",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
        initializer=initializers,
    )


class TestIRVersion(unittest.TestCase):
    def test_older_default_opsets(self):
        for opset, expected_ir in [(7, 3), (8, 4), (9, 4), (11, 6), (15, 8)]:
            with self.subTest(opset=opset):
                model = make_model_ex(identity_graph(), [("", opset)], opset)
                self.assertEqual(model.ir_version, expected_ir)
                checker.check_model(model)

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 23, "ONNX does not support opset 23")
    def test_opset_23_requires_ir_11(self):
        model = make_model_ex(identity_graph(), [("", 23)], 23)
        self.assertEqual(model.ir_version, 11)
        checker.check_model(model)

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 24, "ONNX does not support opset 24")
    def test_opset_24_requires_ir_12(self):
        model = make_model_ex(identity_graph(), [("", 24)], 24)
        self.assertEqual(model.ir_version, 12)
        checker.check_model(model)

    def test_all_onnx_domains_contribute(self):
        for domain, version, expected_ir in [
            ("ai.onnx.ml", 2, 6),
            ("ai.onnx.ml", 3, 8),
            ("ai.onnx.training", 1, 7),
            ("ai.onnx.preview.training", 1, 7),
        ]:
            with self.subTest(domain=domain, version=version):
                model = make_model_ex(identity_graph(), [("", 9), (domain, version)], 9)
                self.assertEqual(model.ir_version, expected_ir)
                checker.check_model(model)

    def test_default_domain_alias(self):
        model = make_model_ex(identity_graph(), [("ai.onnx", 11)], 15)
        self.assertEqual(model.ir_version, 6)

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 21, "ONNX does not support ML opset 5")
    def test_ml_opset_5_requires_ir_10(self):
        model = make_model_ex(identity_graph(), [("", 9), ("ai.onnx.ml", 5)], 15)
        self.assertEqual(model.ir_version, 10)
        checker.check_model(model)

    def test_ml_only_uses_actual_import(self):
        graph = helper.make_graph(
            [helper.make_node("Scaler", ["X"], ["Y"], domain="ai.onnx.ml")],
            "ml_only",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        model = make_model_ex(graph, [("ai.onnx.ml", 2)], 15)
        self.assertEqual(model.ir_version, 6)
        self.assertEqual(
            [(o.domain, o.version) for o in model.opset_import], [("ai.onnx.ml", 2)]
        )
        checker.check_model(model)

    def test_duplicate_imports_use_highest_version(self):
        model = make_model_ex(
            identity_graph(), [("", 9), ("ai.onnx.ml", 1), ("ai.onnx.ml", 3)], 15
        )
        self.assertEqual(model.ir_version, 8)
        self.assertEqual(
            [(o.domain, o.version) for o in model.opset_import],
            [("", 9), ("ai.onnx.ml", 3)],
        )
        checker.check_model(model)

    def test_custom_domain_does_not_force_latest_ir(self):
        model = make_model_ex(identity_graph(), [("", 9), ("com.microsoft", 1)], 15)
        self.assertEqual(model.ir_version, 4)
        self.assertIn(
            ("com.microsoft", 1), [(o.domain, o.version) for o in model.opset_import]
        )
        checker.check_model(model)

    def test_unknown_onnx_opsets_are_rejected(self):
        for domain in ("", "ai.onnx", "ai.onnx.ml", "ai.onnx.training"):
            with self.subTest(domain=domain):
                with self.assertRaisesRegex(ValueError, "Unsupported opset-version"):
                    make_model_ex(identity_graph(), [(domain, 9999)], 9999)

    def test_initializer_without_graph_input_still_requires_ir_4(self):
        # make_model_ex upgrades the requested import from 1 to 8. ONNX's
        # opset-only minimum for 8 is IR 3, but this graph requires IR 4.
        model = make_model_ex(identity_graph(initializer=True), [("", 1)], 15)
        self.assertEqual(model.ir_version, 4)
        self.assertEqual(model.opset_import[0].version, 8)
        checker.check_model(model)
        session = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(session.run(None, {})[0], [1.0, 2.0])

    def test_mixed_domain_model_runs_on_supported_runtime(self):
        # The CI matrix includes ONNX Runtime 1.17.3. Newer ONNX installations
        # must not force their latest IR version onto this older-opset model.
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["X"], ["intermediate"]),
                helper.make_node(
                    "Scaler",
                    ["intermediate"],
                    ["Y"],
                    domain="ai.onnx.ml",
                    scale=[2.0, 3.0],
                    offset=[1.0, -1.0],
                ),
            ],
            "mixed_domains",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])],
        )
        model = make_model_ex(graph, [("", 9), ("ai.onnx.ml", 2)], 15)
        self.assertEqual(model.ir_version, 6)
        checker.check_model(model)
        session = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            session.run(None, {"X": values})[0], [[0.0, 9.0], [4.0, 15.0]]
        )


if __name__ == "__main__":
    unittest.main()
