"""
Tests scikit-dictvectorizer converter.
"""
import unittest
from sklearn.feature_extraction import DictVectorizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import DictionaryType, StringTensorType, FloatTensorType


class TestSklearnDictVectorizerConverter(unittest.TestCase):

    def test_model_dict_vectorizer(self):
        model = DictVectorizer()
        model.fit_transform([{'amy': 1, 'chin': 200}, {'nice': 3, 'amy': 1}])
        model_onnx = convert_sklearn(model, 'dictionary vectorizer',
                                     [('input', DictionaryType(StringTensorType([1]), FloatTensorType([1])))])
        self.assertTrue(model_onnx is not None)

