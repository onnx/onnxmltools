# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML DictVectorizer converter.
"""
import sys
from distutils.version import StrictVersion
import unittest
import onnx
import sklearn
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from sklearn.feature_extraction import DictVectorizer
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLDictVectorizerConverter(unittest.TestCase):

    @unittest.skipIf(
        StrictVersion(coremltools.__version__) > StrictVersion("3.1"),
        reason="untested")
    def test_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{'amy': 1., 'chin': 200.}, {'nice': 3., 'amy': 1.}]
        model.fit_transform(data)
        try:
            model_coreml = coremltools.converters.sklearn.convert(model)
        except NameError as e:
            raise AssertionError(
                "Unable to use coremltools, coremltools.__version__=%r, "
                "onnx.__version__=%r, sklearn.__version__=%r, "
                "sys.platform=%r." % (
                    coremltools.__version__, onnx.__version__,
                    sklearn.__version__, sys.platform)) from e
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="CmlDictVectorizer-OneOff-SkipDim1",
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")


if __name__ == "__main__":
    unittest.main()
