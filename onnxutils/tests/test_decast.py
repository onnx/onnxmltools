import unittest

from onnx import helper
from onnx import onnx_pb as onnx_proto
from onnxconverter_common.decast import decast


class DecastTestCase(unittest.TestCase):

    def test_decast(self):
        nodes = []
        nodes[0:] = [helper.make_node('Identity', ['input1'], ['identity1'])]
        nodes[1:] = [helper.make_node('Cast', ['identity1'], ['cast0'], to=1)]
        nodes[2:] = [helper.make_node('ReduceSum', ['cast0'], ['reduce0'])]
        nodes[3:] = [helper.make_node('Cast', ['reduce0'], ['cast1'], to=6)]
        nodes[4:] = [helper.make_node('Identity', ['cast1'], ['output0'])]

        input0 = helper.make_tensor_value_info('input1', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])
        output0 = helper.make_tensor_value_info('output0', onnx_proto.TensorProto.FLOAT, [1, 1, 2, 3])

        graph = helper.make_graph(nodes, 'test_graph', [input0], [output0])
        model = helper.make_model(graph)
        self.assertIsNotNone(model)

        converted_model = decast(model, ['ReduceSum'])
        self.assertTrue(len(converted_model.graph.node) == 3)


if __name__ == '__main__':
    unittest.main()
