"""
Tests CoreML Scaler converter.
"""
import coremltools
import unittest
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert.coreml.ScalerConverter import ScalerConverter
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLScalerConverter(unittest.TestCase):

    def test_scaler(self):
        model = StandardScaler()
        model.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)