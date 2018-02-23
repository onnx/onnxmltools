"""
Tests scikit-dictvectorizer converter.
"""
import unittest
from sklearn.feature_extraction import DictVectorizer
from onnxmltools.convert.sklearn.DictVectorizerConverter import DictVectorizerConverter
from onnxmltools.convert.sklearn.convert import convert
from onnxmltools.convert.common.ConvertContext import ConvertContext


class TestSklearnDictVectorizerConverter(unittest.TestCase):

    def test_model_dict_vectorizer(self):
        model = DictVectorizer()
        model.fit_transform([{'amy': 1, 'chin': 200}, {'nice': 3, 'amy': 1}])
        model_onnx = convert(model)
        self.assertTrue(model_onnx is not None)

    def test_dict_vectorizer_converter(self):
        model = DictVectorizer()
        model.fit_transform([{'amy': 1, 'chin': 200}, {'nice': 3, 'amy': 1}])

        context = ConvertContext()
        node = DictVectorizerConverter.convert(context, model, ["Input"])
        self.assertTrue(node is not None)
