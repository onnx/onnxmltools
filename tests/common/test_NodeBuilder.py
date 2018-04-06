"""
Tests for NodeBuilder.
"""
import unittest
from onnxmltools.proto import onnx_proto
from onnxmltools.convert.common import NodeBuilder
from onnxmltools.convert.common import model_util
from onnxmltools.convert.common.ConvertContext import ConvertContext

class TestNodeBuilder(unittest.TestCase):

    def test_initializer(self):
        context = ConvertContext()
        nb = NodeBuilder(context, "bar")
        nb.add_input("Input")
        nb.add_output("Output")

        test_array = [1,2,3]
        tensor = model_util.make_tensor('classes', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        nb.add_initializer(tensor)
        node = nb.make_node()

        self.assertEqual(len(node.initializers), 1)
        self.assertEqual(node.initializers[0].name, 'bar_classes')

    def test_multiple_initializers(self):
        context = ConvertContext()
        nb = NodeBuilder(context, "bar")
        nb.add_input("Input")
        nb.add_output("Output")

        test_array = [1,2,3]
        tensor1 = model_util.make_tensor('classes1', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        tensor2 = model_util.make_tensor('classes2', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        tensor3 = model_util.make_tensor('classes3', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        tensor4 = model_util.make_tensor('classes4', onnx_proto.TensorProto.FLOAT, [1,len(test_array)], test_array)
        nb.add_initializer(tensor1)
        nb.add_initializer(tensor2)
        nb.add_initializer(tensor3)
        nb.add_initializer(tensor4)
        node = nb.make_node()

        self.assertEqual(len(node.initializers), 4)
        self.assertEqual(node.initializers[0].name, 'bar_classes1')
        self.assertEqual(node.initializers[1].name, 'bar_classes2')
        self.assertEqual(node.initializers[2].name, 'bar_classes3')
        self.assertEqual(node.initializers[3].name, 'bar_classes4')


    def test_value(self):
        context = ConvertContext()
        nb = NodeBuilder(context, "bar")
        nb.add_input("Input")
        nb.add_output("Output")

        test_array = [1,2,3]
        value = model_util.make_tensor('value_test', onnx_proto.TensorProto.FLOAT, [1, len(test_array)], test_array)
        nb.add_value(value)
        node = nb.make_node()

        self.assertEqual(len(node.values), 1)
        self.assertEqual(node.values[0].name, 'bar_value_test')

    def test_add_inputs(self):
        context = ConvertContext()
        nb = NodeBuilder(context, "foo")
        nb.add_input('test')
        nb.add_empty_input()
        nb.add_input(model_util.make_tensor_value_info('value_test', onnx_proto.TensorProto.FLOAT, [1, 3]))

        test_array = [1,2,3]
        init = model_util.make_tensor('init', onnx_proto.TensorProto.FLOAT, [1, len(test_array)], test_array)
        nb.add_initializer(init)

        value = model_util.make_tensor('value', onnx_proto.TensorProto.FLOAT, [1, len(test_array)], test_array)
        nb.add_value(value)
        node = nb.make_node()

        input_names = node.input_names
        self.assertEqual(len(input_names),5)

        # Confirm the order of the names based upon when added
        expected_names = ['test','','value_test','foo_init', 'foo_value']
        self.assertEqual(input_names, expected_names)
