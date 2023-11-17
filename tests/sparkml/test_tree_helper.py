# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from onnx import TensorProto
from onnx.defs import onnx_opset_version
from onnx.helper import (
    make_node,
    make_graph,
    make_model,
    make_tensor_value_info,
    make_opsetid,
)
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxruntime import InferenceSession
from onnxmltools.convert.sparkml.operator_converters.tree_helper import Node

TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmDecisionTreeClassifierBig(unittest.TestCase):
    @unittest.skipIf(TARGET_OPSET < 17, reason="Opset 17 is needed")
    def test_split(self):
        attrs = {
            "class_ids": [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ],
            "class_nodeids": [
                3,
                3,
                3,
                4,
                4,
                4,
                6,
                6,
                6,
                7,
                7,
                7,
                10,
                10,
                10,
                11,
                11,
                11,
                12,
                12,
                12,
            ],
            "class_treeids": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "class_weights": [
                0.4,
                0.6,
                0.0,
                0.95,
                0.04,
                0.0,
                0.0,
                0.185,
                0.814,
                0.372,
                0.62,
                0.0,
                0.74,
                0.21,
                0.03,
                0.0,
                1.0,
                0.0,
                0.87,
                0.05,
                0.071,
            ],
            "classlabels_int64s": [0, 1, 2],
            "name": "TreeEnsembleClassifier",
            "nodes_falsenodeids": [8, 5, 4, 0, 0, 7, 0, 0, 12, 11, 0, 0, 0],
            "nodes_featureids": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "nodes_hitrates": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "nodes_missing_value_tracks_true": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "nodes_modes": [
                "BRANCH_LEQ",
                "||",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "LEAF",
            ],
            "nodes_nodeids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "nodes_treeids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nodes_truenodeids": [1, 2, 3, 0, 0, 6, 0, 0, 9, 10, 0, 0, 0],
            "nodes_values": [
                10.5,
                [55, 59, 65],
                1.5,
                0.0,
                0.0,
                8.5,
                0.0,
                0.0,
                10.5,
                9.5,
                0.0,
                0.0,
                0.0,
            ],
            "post_transform": None,
        }
        root, nodes = Node.create(attrs)
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
        n_nodes1 = len(list(root))
        root.unfold_rule_or()
        n_nodes2 = len(list(root))
        if n_nodes1 >= n_nodes2:
            raise AssertionError(f"Unexpected {n_nodes1} >= {n_nodes2}")
        ns = {}
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
            if key in ns:
                raise AssertionError(f"Duplicate node id {key}.")
            ns[key] = node

        # back to onnx
        new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        for k in attrs:
            if k in {"post_transform"}:
                continue
            if len(attrs[k]) > len(new_attrs[k]):
                raise AssertionError(
                    f"Issue with attribute {k!r}\n"
                    f"{attrs['nodes_modes']}\nbefore {attrs[k]!r}"
                    f"\nafter  {new_attrs[k]!r}\n{new_attrs['nodes_modes']}"
                )

        node = make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["L", "Y"],
            **new_attrs,
        )
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        L = make_tensor_value_info("L", TensorProto.INT64, [None])
        graph = make_graph([node], "n", [X], [L, Y])
        opset_imports = [make_opsetid("", 17), make_opsetid("ai.onnx.ml", 3)]
        model = make_model(graph, opset_imports=opset_imports)
        sess = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        x = np.arange(20).reshape((-1, 2)).astype(np.float32)
        got = sess.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].tolist(), [2, 2, 2, 2, 2, 1, 0, 0, 0, 0])

        # again
        root, nodes = Node.create(new_attrs)
        root.unfold_rule_or()
        new_new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        self.assertEqual(new_attrs, new_new_attrs)

    @unittest.skipIf(TARGET_OPSET < 17, reason="Opset 17 is needed")
    def test_split_non_contiguous_ids(self):
        attrs = {
            "class_ids": [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ],
            "class_nodeids": [
                3,
                3,
                3,
                4,
                4,
                4,
                6,
                6,
                6,
                7,
                7,
                7,
                10,
                10,
                10,
                11,
                11,
                11,
                13,
                13,
                13,
            ],
            "class_treeids": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "class_weights": [
                0.4,
                0.6,
                0.0,
                0.95,
                0.04,
                0.0,
                0.0,
                0.185,
                0.814,
                0.372,
                0.62,
                0.0,
                0.74,
                0.21,
                0.03,
                0.0,
                1.0,
                0.0,
                0.87,
                0.05,
                0.071,
            ],
            "classlabels_int64s": [0, 1, 2],
            "name": "TreeEnsembleClassifier",
            "nodes_falsenodeids": [8, 5, 4, 0, 0, 7, 0, 0, 13, 11, 0, 0, 0],
            "nodes_featureids": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "nodes_hitrates": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "nodes_missing_value_tracks_true": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "nodes_modes": [
                "BRANCH_LEQ",
                "||",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "LEAF",
            ],
            "nodes_nodeids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
            "nodes_treeids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nodes_truenodeids": [1, 2, 3, 0, 0, 6, 0, 0, 9, 10, 0, 0, 0],
            "nodes_values": [
                10.5,
                [55, 59, 65],
                1.5,
                0.0,
                0.0,
                8.5,
                0.0,
                0.0,
                10.5,
                9.5,
                0.0,
                0.0,
                0.0,
            ],
            "post_transform": None,
        }
        root, nodes = Node.create(attrs)
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
        n_nodes1 = len(list(root))
        root.unfold_rule_or()
        n_nodes2 = len(list(root))
        if n_nodes1 >= n_nodes2:
            raise AssertionError(f"Unexpected {n_nodes1} >= {n_nodes2}")
        ns = {}
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
            if key in ns:
                raise AssertionError(f"Duplicate node id {key}.")
            ns[key] = node

        # back to onnx
        new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        for k in attrs:
            if k in {"post_transform"}:
                continue
            if len(attrs[k]) > len(new_attrs[k]):
                raise AssertionError(
                    f"Issue with attribute {k!r}\n"
                    f"{attrs['nodes_modes']}\nbefore {attrs[k]!r}"
                    f"\nafter  {new_attrs[k]!r}\n{new_attrs['nodes_modes']}"
                )

        node = make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["L", "Y"],
            **new_attrs,
        )
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        L = make_tensor_value_info("L", TensorProto.INT64, [None])
        graph = make_graph([node], "n", [X], [L, Y])
        opset_imports = [make_opsetid("", 17), make_opsetid("ai.onnx.ml", 3)]
        model = make_model(graph, opset_imports=opset_imports)
        sess = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        x = np.arange(20).reshape((-1, 2)).astype(np.float32)
        got = sess.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].tolist(), [2, 2, 2, 2, 2, 1, 0, 0, 0, 0])

        # again
        root, nodes = Node.create(new_attrs)
        root.unfold_rule_or()
        new_new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        self.assertEqual(new_attrs, new_new_attrs)

    @unittest.skipIf(TARGET_OPSET < 17, reason="Opset 17 is needed")
    def test_split_more_complex(self):
        attrs = {
            "class_ids": [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
            ],
            "class_nodeids": [
                3,
                3,
                3,
                4,
                4,
                4,
                6,
                6,
                6,
                7,
                7,
                7,
                10,
                10,
                10,
                11,
                11,
                11,
                12,
                12,
                12,
            ],
            "class_treeids": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "class_weights": [
                0.4,
                0.6,
                0.0,
                0.95,
                0.04,
                0.0,
                0.0,
                0.185,
                0.814,
                0.372,
                0.62,
                0.0,
                0.74,
                0.21,
                0.03,
                0.0,
                1.0,
                0.0,
                0.87,
                0.05,
                0.071,
            ],
            "classlabels_int64s": [0, 1, 2],
            "name": "TreeEnsembleClassifier",
            "nodes_falsenodeids": [8, 5, 4, 0, 0, 7, 0, 0, 12, 11, 0, 0, 0],
            "nodes_featureids": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "nodes_hitrates": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "nodes_missing_value_tracks_true": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "nodes_modes": [
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "BRANCH_LEQ",
                "BRANCH_LEQ",
                "LEAF",
                "LEAF",
                "LEAF",
            ],
            "nodes_nodeids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "nodes_treeids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "nodes_truenodeids": [1, 2, 3, 0, 0, 6, 0, 0, 9, 10, 0, 0, 0],
            "nodes_values": [
                [100, 101, 102],
                [55, 59, 65],
                [10, 11, 12],
                0.0,
                0.0,
                [1000, 1001, 1002],
                0.0,
                0.0,
                [10000, 10001],
                [100000, 100001],
                0.0,
                0.0,
                0.0,
            ],
            "post_transform": None,
        }
        root, nodes = Node.create(attrs)
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
        n_nodes1 = len(list(root))
        root.unfold_rule_or()
        n_nodes2 = len(list(root))
        if n_nodes1 >= n_nodes2:
            raise AssertionError(f"Unexpected {n_nodes1} >= {n_nodes2}")
        ns = {}
        for node in root:
            key = node.nodes_treeids, node.nodes_nodeids
            if None in key:
                raise AssertionError(f"Wrong key for node {node!r}.")
            if key in ns:
                raise AssertionError(f"Duplicate node id {key}.")
            ns[key] = node

        # back to onnx
        new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        for k in attrs:
            if k in {"post_transform"}:
                continue
            if len(attrs[k]) > len(new_attrs[k]):
                raise AssertionError(
                    f"Issue with attribute {k!r}\n{attrs['nodes_modes']}"
                    f"\nbefore {attrs[k]!r}"
                    f"\nafter  {new_attrs[k]!r}\n{new_attrs['nodes_modes']}"
                )

        self.assertEqual(len(new_attrs["nodes_modes"]), len(attrs["nodes_modes"]) + 10)
        node = make_node(
            op_type="TreeEnsembleClassifier",
            inputs=["X"],
            outputs=["L", "Y"],
            **new_attrs,
        )
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        L = make_tensor_value_info("L", TensorProto.INT64, [None])
        graph = make_graph([node], "n", [X], [L, Y])
        opset_imports = [make_opsetid("", 17), make_opsetid("ai.onnx.ml", 3)]
        model = make_model(graph, opset_imports=opset_imports)
        sess = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        x = np.arange(20).reshape((-1, 2)).astype(np.float32)
        got = sess.run(None, {"X": x})
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # again
        root, nodes = Node.create(new_attrs)
        root.unfold_rule_or()
        new_new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        self.assertEqual(new_attrs, new_new_attrs)

    def test_debug(self):
        try:
            from onnxmltools.debug1 import attrs
        except ImportError:
            return
        root, nodes = Node.create(attrs)
        root.unfold_rule_or()
        new_attrs = root.to_attrs(
            post_transform=None,
            name="TreeEnsembleClassifier",
            classlabels_int64s=[0, 1, 2],
            domain="ai.onnx.ml",
        )
        for k in attrs:
            if k in {"post_transform"}:
                continue
            if len(attrs[k]) > len(new_attrs[k]):
                raise AssertionError(
                    f"Issue with attribute {k!r}\n{len(new_attrs[k])}"
                    f"\nbefore {len(attrs[k])}."
                )


if __name__ == "__main__":
    # TestSparkmDecisionTreeClassifierBig().test_debug()
    unittest.main(verbosity=2)
