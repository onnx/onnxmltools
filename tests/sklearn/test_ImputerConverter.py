"""
Tests scikit-imputer converter.
"""
import unittest
import numpy as np
from sklearn.preprocessing import Imputer
from onnxmltools.convert.sklearn.ImputerConverter import ImputerConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext
from onnxmltools.convert.common.model_util import make_tensor_value_info
from onnxmltools.proto import onnx_proto


class TestSklearnImputerConverter(unittest.TestCase):

    def test_model_imputer(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])
        model_onnx = convert(model, 'scikit-learn imputer', [('features', 'int32', 2)])
        self.assertTrue(model_onnx is not None)

    def test_imputer_int_inputs(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])

        context = ConvertContext()
        node = ImputerConverter.convert(context, model,
            [make_tensor_value_info('features', onnx_proto.TensorProto.INT32, [2])])
        self.assertTrue(node is not None)

        # should contain two nodes
        self.assertEqual(len(node), 2)
        # last node should contain the Imputer
        outputs = node[-1].outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value, 2)

    def test_imputer_float_inputs(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])

        context = ConvertContext()
        node = ImputerConverter.convert(context, model,
            [make_tensor_value_info('features', onnx_proto.TensorProto.FLOAT, [2])])
        self.assertTrue(node is not None)

        # should contain two nodes
        self.assertEqual(len(node), 1)

        # last node should contain the Imputer
        outputs = node[-1].outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value, 2)
