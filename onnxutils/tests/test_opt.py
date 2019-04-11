import os
import unittest
import numpy as np

import onnx
from onnx import helper
from onnx import onnx_pb as onnx_proto
from onnxconverter_common.optimizer import optimize_onnx, optimize_onnx_model

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')



class OptimizerTestCase(unittest.TestCase):
    @staticmethod
    def get_temp_file(name):
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        return os.path.join(tmp_path, name)

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
        onnx.save_model(model, self.get_temp_file('temp_before.onnx'))
        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        new_nodes = [n_ for n_ in new_nodes if not isinstance(n_, tuple)]
        self.assertEqual(len(new_nodes), 3)
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        onnx.save_model(model, self.get_temp_file('temp_after.onnx'))
        self.assertIsNotNone(model)

    def test_move_transpose(self):
        val = np.asarray([[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]], np.float32)

        nodes = []
        nodes[0:] = \
            [helper.make_node('Constant', [], ['const1'], value=helper.make_tensor(
                name='const0',
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=val.shape,
                vals=val.flatten().astype(float)))]
        nodes[1:] = [helper.make_node('Identity', ['const1'], ['identity1'])]
        nodes[2:] = [helper.make_node('Identity', ['identity1'], ['identity2'])]
        nodes[3:] = [helper.make_node('Max', ['input1', 'identity2'], ['max0'])]
        nodes[4:] = [helper.make_node('Transpose', ['max0'], ['tranpose0'], perm=[0, 2, 3, 1])]
        nodes[5:] = [helper.make_node('LeakyRelu', ['tranpose0'], ['tranpose1'])]
        nodes[6:] = [helper.make_node('Relu', ['tranpose1'], ['output0'], perm=(0, 3, 1, 2))]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        onnx.save_model(model, self.get_temp_file('temp_before.onnx'))
        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        new_nodes = [n_ for n_ in new_nodes if not isinstance(n_, tuple)]
        self.assertEqual(len(new_nodes), 5)
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        onnx.save_model(model, self.get_temp_file('temp_after.onnx'))
        self.assertIsNotNone(model)

    def test_merge(self):
        val = np.asarray([[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]], np.float32)

        nodes = []
        nodes[0:] = \
            [helper.make_node('Constant', [], ['const1'], value=helper.make_tensor(
                name='const0',
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=val.shape,
                vals=val.flatten().astype(float)))]
        nodes[1:] = [helper.make_node('Max', ['input1'], ['max0'])]
        nodes[2:] = [helper.make_node('Transpose', ['max0'], ['tranpose0'], perm=[0, 2, 3, 1])]
        nodes[3:] = [helper.make_node('Transpose', ['tranpose0'], ['add_input1'], perm=(0, 3, 1, 2))]
        nodes[4:] = [helper.make_node('Add', ['max0', 'add_input1'], ['output0'])]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        onnx.save_model(model, self.get_temp_file('temp_before.onnx'))
        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        new_nodes = [n_ for n_ in new_nodes if not isinstance(n_, tuple)]
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        onnx.save_model(model, self.get_temp_file('temp_after.onnx'))
        self.assertEqual(len(new_nodes), 3)
        self.assertIsNotNone(model)

    def test_fan_out(self):
        val = np.asarray([[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]], np.float32)

        nodes = []
        nodes[0:] = \
            [helper.make_node('Constant', [], ['const1'], value=helper.make_tensor(
                name='const0',
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=val.shape,
                vals=val.flatten().astype(float)),
                name="0")]
        nodes[1:] = [helper.make_node('Identity', ['const1'], ['identity1'], name="1")]
        nodes[2:] = [helper.make_node('Identity', ['identity1'], ['identity2'], name="2")]
        nodes[3:] = [helper.make_node('Max', ['input1', 'identity2'], ['max0'], name="3")]
        nodes[4:] = [helper.make_node('Transpose', ['max0'], ['tranpose0'], perm=[0, 2, 3, 1], name="4")]
        nodes[5:] = [helper.make_node('LeakyRelu', ['tranpose0'], ['leak0'], name="5")]
        nodes[6:] = [helper.make_node('LeakyRelu', ['leak0'], ['leak1'], name="6")]
        nodes[7:] = [helper.make_node('Transpose', ['leak1'], ['add_input1'], perm=(0, 3, 1, 2), name="7")]
        nodes[8:] = [helper.make_node('Add', ['leak0', 'add_input1'], ['output0'], name="8")]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        onnx.save_model(model, self.get_temp_file('temp_before.onnx'))
        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        new_nodes = [n_ for n_ in new_nodes if not isinstance(n_, tuple)]
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        onnx.save_model(model, self.get_temp_file('temp_after.onnx'))
        self.assertEqual(len(new_nodes), 6)
        self.assertIsNotNone(model)

    def test_fan_in(self):
        val = np.asarray([[[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]]], np.float32)

        nodes = []
        nodes[0:] = \
            [helper.make_node('Constant', [], ['const1'], value=helper.make_tensor(
                name='const0',
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=val.shape,
                vals=val.flatten().astype(float)),
                name="0")]
        nodes[1:] = [helper.make_node('Identity', ['const1'], ['identity1'], name="1")]
        nodes[2:] = [helper.make_node('Identity', ['identity1'], ['identity2'], name="2")]
        nodes[3:] = [helper.make_node('Max', ['input1', 'identity2'], ['max0'], name="3")]
        nodes[4:] = [helper.make_node('LeakyRelu', ['max0'], ['leak0'], name="4")]
        nodes[5:] = [helper.make_node('LeakyRelu', ['leak0'], ['leak1'], name="5")]
        nodes[6:] = [helper.make_node('LeakyRelu', ['leak0'], ['leak2'], name="6")]
        nodes[7:] = [helper.make_node('Transpose', ['leak1'], ['tranpose0'], perm=[0, 2, 3, 1], name="7")]
        nodes[8:] = [helper.make_node('Transpose', ['leak2'], ['tranpose1'], perm=[0, 2, 3, 1], name="8")]
        nodes[9:] = [helper.make_node('Add', ['tranpose0', 'tranpose1'], ['add0'], name="9")]
        nodes[10:] = [helper.make_node('Transpose', ['add0'], ['tranpose2'], perm=[0, 3, 1, 2], name="10")]
        nodes[11:] = [helper.make_node('Conv', ['tranpose2'], ['output0'], name="11")]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        onnx.save_model(model, self.get_temp_file('temp_before.onnx'))
        new_nodes = optimize_onnx(nodes, inputs=[input0], outputs=[output0])
        new_nodes = [n_ for n_ in new_nodes if not isinstance(n_, tuple)]
        graph = helper.make_graph(new_nodes, 'test0', [input0], [output0])
        model = helper.make_model(graph)
        onnx.save_model(model, self.get_temp_file('temp_after.onnx'))
        self.assertEqual(len(new_nodes), 7)
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
