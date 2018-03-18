"""
Tests CoreML Imputer converter.
"""
import coremltools
import numpy as np
import unittest
from sklearn.preprocessing import Imputer
from onnxmltools.convert.coreml.convert import convert


class TestCoreMLImputerConverter(unittest.TestCase):

    def test_imputer(self):
        model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        model.fit([[1, 2], [np.nan, 3], [7, 6]])
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
