"""
Tests CoreML DictVectorizer converter.
"""
import coremltools
import unittest
from sklearn.feature_extraction import DictVectorizer
from onnxmltools.convert.coreml.DictVectorizerConverter import DictVectorizerConverter
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLDictVectorizerConverter(unittest.TestCase):

    def test_dict_vectorizer(self):
        model = DictVectorizer()
        model.fit_transform([{'amy': 1, 'chin': 200}, {'nice': 3, 'amy': 1}])
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)