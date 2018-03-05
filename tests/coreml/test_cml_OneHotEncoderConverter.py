"""
Main functions to convert machine learned model from *Core ML* model to *ONNX*.
"""
import sys
import os
import unittest
import coremltools
from onnxmltools.convert.coreml.convert import convert


class TestCoremlOneHotEncoderConverter(unittest.TestCase):

    def test_one_hot_encoder(self):
        script_dir = os.path.dirname(__file__)
        relative_path = "../data/onehot_simple.mlmodel"
        abs_file = os.path.join(script_dir, relative_path)
        model_coreml = coremltools.utils.load_spec(abs_file)
        model_onnx = convert(model_coreml)
        self.assertTrue(model_onnx is not None)
