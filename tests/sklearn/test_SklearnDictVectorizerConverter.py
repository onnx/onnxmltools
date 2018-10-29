"""
Tests scikit-dictvectorizer converter.
"""
import unittest
from sklearn.feature_extraction import DictVectorizer
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import DictionaryType, StringTensorType, FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnDictVectorizerConverter(unittest.TestCase):

    def test_model_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{'amy': 1., 'chin': 200.}, {'nice': 3., 'amy': 1.}]
        model.fit_transform(data)
        model_onnx = convert_sklearn(model, 'dictionary vectorizer',
                                     [('input', DictionaryType(StringTensorType([1]), FloatTensorType([1])))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnDictVectorizer-OneOff-SkipDim1")


if __name__ == "__main__":
    unittest.main()

