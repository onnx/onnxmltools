'''
Tests Model Builder
'''
import unittest
from onnxmltools.proto import onnx_proto
from onnxmltools.convert.common import NodeBuilder
from onnxmltools.convert.common import ModelBuilder
from onnxmltools.convert.common import model_util
from onnxmltools.convert.common import ConvertContext

class TestModelBuilder(unittest.TestCase):
    def test_initializers(self):

        context = ConvertContext()
        # create nodes with initializers
        mb = ModelBuilder()
        nb = NodeBuilder(context, 'bar')
        nb.add_input('Input')
        nb.add_output('Output')

        test_array = [1,2,3]
        tensor = model_util.make_tensor('classes', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        nb.add_initializer(tensor)
        node = nb.make_node()

        mb.add_nodes([node.onnx_node])
        mb.add_initializers(node.initializers)
        mb.add_inputs([model_util.make_tensor_value_info('Input', onnx_proto.TensorProto.FLOAT, [1])])
        mb.add_outputs([model_util.make_tensor_value_info('Output', onnx_proto.TensorProto.FLOAT, [1])])
        model = mb.make_model()
        self.assertEqual(len(model.graph.initializer), 1)
        self.assertEqual(model.graph.initializer[0].name, 'bar.classes')

    def test_intitializers_on_multiple_nodes(self):
        context = ConvertContext()
        mb = ModelBuilder()
        nb = NodeBuilder(context, 'bar')
        nb.add_input('Input')
        nb.add_output('Output')

        test_array = [1,2,3]
        tensor = model_util.make_tensor('classes', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        nb.add_initializer(tensor)
        node = nb.make_node()

        nb2 = NodeBuilder(context, 'bar2')
        nb2.add_input('Output')
        nb2.add_output('Output2')
        tensor2 = model_util.make_tensor('classes2', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        nb2.add_initializer(tensor2)
        node2 = nb2.make_node()

        mb.add_nodes([node.onnx_node, node2.onnx_node])
        mb.add_initializers(node.initializers)
        mb.add_initializers(node2.initializers)
        mb.add_inputs([model_util.make_tensor_value_info('Input', onnx_proto.TensorProto.FLOAT, [1])])
        mb.add_outputs([model_util.make_tensor_value_info('Output', onnx_proto.TensorProto.FLOAT, [1])])
        model = mb.make_model()
        self.assertEqual(len(model.graph.initializer), 2)
        self.assertEqual(model.graph.initializer[0].name, 'bar.classes')
        self.assertEqual(model.graph.initializer[1].name, 'bar2.classes2')

