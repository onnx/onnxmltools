"""
Tests scikit-imputer converter.
"""
import unittest
import numpy as np
from sklearn.preprocessing import Imputer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType


class TestSklearnImputerConverter(unittest.TestCase):

    def test_model_imputer(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])
        model_onnx = convert_sklearn(model, 'scikit-learn imputer', [('input', Int64TensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)

    def test_imputer_int_inputs(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])

        model_onnx = convert_sklearn(model, 'scikit-learn imputer', [('input', Int64TensorType([1, 2]))])
        self.assertEqual(len(model_onnx.graph.node), 2)

        # Last node should be Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)

        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value, 2)

    def test_imputer_float_inputs(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])

        model_onnx = convert_sklearn(model, 'scikit-learn imputer', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx.graph.node is not None)

        # should contain only node
        self.assertEqual(len(model_onnx.graph.node), 1)

        # last node should contain the Imputer
        outputs = model_onnx.graph.output
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type.tensor_type.shape.dim[-1].dim_value, 2)
