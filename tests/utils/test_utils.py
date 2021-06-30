# SPDX-License-Identifier: Apache-2.0

"""
Tests utilities.
"""
import os
import unittest
import warnings
import onnxmltools
from onnxmltools.utils import load_model, save_model
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
        os.remove(new_onnx_file)

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


class TestWrapper(unittest.TestCase):

    @unittest.skipIf(True, reason="Needs this PR: https://github.com/onnx/tensorflow-onnx/pull/1563")
    def test_keras_with_tf2onnx(self):
        import tensorflow.keras as keras
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=4, input_shape=(10,), activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
        onnx_model = onnxmltools.convert_tensorflow(model)
        self.assertTrue(len(onnx_model.graph.node) > 0)


if __name__ == "__main__":
    unittest.main()
