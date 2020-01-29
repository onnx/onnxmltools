"""
Tests utilities.
"""
import os
import six
import unittest
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


@unittest.skipIf(six.PY2, "Keras and Tensorflow converter not support python 2.x")
class TestWrapper(unittest.TestCase):

    def test_keras_with_tf2onnx(self):
        import keras2onnx
        from keras2onnx.proto import keras
        from keras2onnx.proto.tfcompat import is_tf2
        if not is_tf2:  # tf2onnx is not available for tensorflow 2.0 yet.
            model = keras.Sequential()
            model.add(keras.layers.Dense(units=4, input_shape=(10,), activation='relu'))
            model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
            graph_def = keras2onnx.export_tf_frozen_graph(model)
            onnx_model = onnxmltools.convert_tensorflow(graph_def, **keras2onnx.build_io_names_tf2onnx(model))
            self.assertTrue(len(onnx_model.graph.node) > 0)


if __name__ == "__main__":
    unittest.main()
