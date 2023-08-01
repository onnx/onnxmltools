# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML DictVectorizer converter.
"""
import sys
import packaging.version as pv
import unittest
import onnx
import sklearn

try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing

    if not hasattr(sklearn.preprocessing, "Imputer"):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, "Imputer", Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from sklearn.feature_extraction import DictVectorizer
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestCoreMLDictVectorizerConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(coremltools.__version__) > pv.Version("3.1"), reason="untested"
    )
    def test_dict_vectorizer(self):
        model = DictVectorizer()
        data = [{"amy": 1.0, "chin": 200.0}, {"nice": 3.0, "amy": 1.0}]
        model.fit_transform(data)
        try:
            model_coreml = coremltools.converters.sklearn.convert(model)
        except NameError as e:
            raise AssertionError(
                "Unable to use coremltools, coremltools.__version__=%r, "
                "onnx.__version__=%r, sklearn.__version__=%r, "
                "sys.platform=%r."
                % (
                    coremltools.__version__,
                    onnx.__version__,
                    sklearn.__version__,
                    sys.platform,
                )
            ) from e
        model_onnx = convert(model_coreml.get_spec(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="CmlDictVectorizer-OneOff-SkipDim1"
        )


if __name__ == "__main__":
    unittest.main()
