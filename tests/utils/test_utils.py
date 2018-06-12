"""
Tests utilities.
"""
import filecmp
import os
import unittest

from onnxmltools.utils import load_model, save_model, save_text
from onnxmltools.utils import set_model_version, set_model_domain, set_model_doc_string


class TestUtils(unittest.TestCase):

    @staticmethod
    def _parseEOL(file):
        with open(file, 'r') as f:
            content = f.read()
        return content.replace("\r\n", "\n")

    def test_load_model(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        self.assertTrue(onnx_model is not None)

    def test_save_model(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        new_onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing2.onnx")
        onnx_model = load_model(onnx_file)

        save_model(onnx_model, new_onnx_file)
        self.assertTrue(os.path.exists(new_onnx_file))

    def test_save_text(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        json_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.json")
        json_file_new = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing_new.json")
        save_text(onnx_model, json_file_new)
        try:
            filecmp.clear_cache()
        except AttributeError:
            # Only available in Python 3
            pass
        content1 = self._parseEOL(json_file)
        content2 = self._parseEOL(json_file_new)
        self.assertTrue(content1 == content2,
                        "Output file from save_text is different than reference output.")

    def test_model_setters(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        set_model_version(onnx_model, 2)
        set_model_domain(onnx_model, "com.sample")
        set_model_doc_string(onnx_model, "Sample docstring")

        self.assertEqual(onnx_model.model_version, 2)
        self.assertEqual(onnx_model.domain, "com.sample")
        self.assertEqual(onnx_model.doc_string, "Sample docstring")

    def test_set_docstring_blank(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "models", "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = load_model(onnx_file)
        set_model_doc_string(onnx_model, "sample")
        self.assertRaises(ValueError, set_model_doc_string, onnx_model.doc_string, "sample")
        set_model_doc_string(onnx_model, "", True)
        self.assertEqual(onnx_model.doc_string, "")
