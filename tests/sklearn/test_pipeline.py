import unittest
from onnxmltools.convert.sklearn.SklearnConvertContext import SklearnConvertContext as ConvertContext
from onnxmltools.convert.sklearn.convert import _combine_inputs
from onnxmltools.convert.common import model_util
from onnxmltools.proto import onnx_proto
from onnxmltools import convert_sklearn

class TestSklearnPipeline(unittest.TestCase):

    def test_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        scaler = StandardScaler()
        scaler.fit([[0, 0],[0, 0],[1, 1],[1, 1]])
        model = Pipeline([('scaler1',scaler),('scaler2', scaler)])

        model_onnx = convert_sklearn(model)
        self.assertTrue(model_onnx is not None)

    def test_combine_inputs(self):
        context = ConvertContext()

        input1 = model_util.make_tensor_value_info("foo", onnx_proto.TensorProto.FLOAT, [1, 3])
        input2 = model_util.make_tensor_value_info("foo2", onnx_proto.TensorProto.FLOAT, [1, 2])
        inputs = (input1, input2)
        nodes = _combine_inputs(context, inputs)

        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(nodes[0].outputs), 1)
        self.assertEqual(nodes[0].outputs[0].type.tensor_type.shape.dim[0].dim_value, 1)
        self.assertEqual(nodes[0].outputs[0].type.tensor_type.shape.dim[1].dim_value, 5)

    def test_combine_inputs_floats_ints(self):
        context = ConvertContext()

        input1 = model_util.make_tensor_value_info("foo", onnx_proto.TensorProto.FLOAT, [1, 3])
        input2 = model_util.make_tensor_value_info("foo2", onnx_proto.TensorProto.INT64, [1, 2])
        input3 = model_util.make_tensor_value_info("foo3", onnx_proto.TensorProto.UINT8, [1, 4])
        inputs = [input1, input2, input3]
        nodes = _combine_inputs(context, inputs)

        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[-1].outputs[0].type.tensor_type.shape.dim[0].dim_value, 1)
        self.assertEqual(nodes[-1].outputs[0].type.tensor_type.shape.dim[1].dim_value, 9)

    def test_combine_inputs_with_string(self):
        context = ConvertContext()

        input1 = model_util.make_tensor_value_info("foo", onnx_proto.TensorProto.FLOAT, [1, 3])
        input2 = model_util.make_tensor_value_info("foo2", onnx_proto.TensorProto.STRING, [1, 2])
        inputs = [input1, input2]
        self.assertRaises(RuntimeError, _combine_inputs, context, inputs)

