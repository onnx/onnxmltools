"""
Tests utilities.
"""
from distutils.version import StrictVersion
import filecmp
import os
import unittest
import numpy as np

from onnxmltools.proto import onnx, onnx_proto, helper, get_opset_number_from_onnx
from onnxmltools.convert.common.optimizer import optimize_onnx
from onnxmltools.utils import load_model, save_model
from onnxmltools.utils import set_denotation, set_model_version, set_model_domain, set_model_doc_string
from onnxmltools.utils.utils_backend import evaluate_condition, is_backend_enabled


class TestUtils(unittest.TestCase):

    @staticmethod
    def _parseEOL(file):
        with open(file, 'r') as f:
            content = f.read()
        return content.replace("\r\n", "\n")

    def test_load_model(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        self.assertTrue(onnx_model is not None)

    def test_save_model(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        new_onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing2.onnx")
        onnx_model = load_model(onnx_file)

        save_model(onnx_model, new_onnx_file)
        self.assertTrue(os.path.exists(new_onnx_file))

    def test_model_setters(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        set_model_version(onnx_model, 2)
        set_model_domain(onnx_model, "com.sample")
        set_model_doc_string(onnx_model, "Sample docstring")

        self.assertEqual(onnx_model.model_version, 2)
        self.assertEqual(onnx_model.domain, "com.sample")
        self.assertEqual(onnx_model.doc_string, "Sample docstring")

    def test_set_docstring_blank(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        set_model_doc_string(onnx_model, "sample")
        self.assertRaises(ValueError, set_model_doc_string, onnx_model.doc_string, "sample")
        set_model_doc_string(onnx_model, "", True)
        self.assertEqual(onnx_model.doc_string, "")

    @unittest.skipIf(StrictVersion(onnx.__version__) < StrictVersion('1.2.1'),
                     "not supported in this ONNX version")
    def test_set_denotation(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        set_denotation(onnx_model, "1", "IMAGE", get_opset_number_from_onnx(), dimension_denotation=["DATA_FEATURE"])
        self.assertEqual(onnx_model.graph.input[0].type.denotation, "IMAGE")
        self.assertEqual(onnx_model.graph.input[0].type.tensor_type.shape.dim[0].denotation, "DATA_FEATURE")

    def test_optimizer(self):
        val = np.asarray([[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]], np.float32)

        nodes = []
        nodes[0:] =\
            [helper.make_node('Constant', [], ['const1'], value=helper.make_tensor(
            name='const0',
            data_type=onnx_proto.TensorProto.FLOAT,
            dims=val.shape,
            vals=val.flatten().astype(float)))]
        nodes[1:] = [helper.make_node('Identity', ['const1'], ['identity1'])]
        nodes[2:] = [helper.make_node('Identity', ['identity1'], ['identity2'])]
        nodes[3:] = [helper.make_node('Max', ['input1', 'identity2'], ['max0'])]
        nodes[4:] = [helper.make_node('Transpose', ['max0'], ['tranpose0'], perm=[0, 2, 3, 1])]
        nodes[5:] = [helper.make_node('Transpose', ['tranpose0'], ['tranpose1'], perm=(0, 3, 1, 2))]
        nodes[6:] = [helper.make_node('Relu', ['tranpose1'], ['output0'], perm=(0, 3, 1, 2))]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        self.assertEqual(len(new_nodes), 3)
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
