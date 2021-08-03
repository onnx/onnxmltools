# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML DictVectorizer converter.
"""
from distutils.version import StrictVersion
import sys
import onnx
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
import unittest
from sklearn.feature_extraction import DictVectorizer
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLDictVectorizerConverter(unittest.TestCase):

    @unittest.skipIf(
        sys.platform == "win32" and
            StrictVersion(coremltools.__version__) <= StrictVersion("3.1") and
            StrictVersion(onnx.__version__) >= StrictVersion("1.9.0"),
        reason="incompabilities scikit-learn, coremltools")
    def test_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{'amy': 1., 'chin': 200.}, {'nice': 3., 'amy': 1.}]
        model.fit_transform(data)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="CmlDictVectorizer-OneOff-SkipDim1",
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")


if __name__ == "__main__":
    unittest.main()
