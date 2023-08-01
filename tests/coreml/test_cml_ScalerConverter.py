# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML Scaler converter.
"""
import unittest
import packaging.version as pv
import numpy

try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing

    if not hasattr(sklearn.preprocessing, "Imputer"):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, "Imputer", Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestCoreMLScalerConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(coremltools.__version__) > pv.Version("3.1"), reason="untested"
    )
    def test_scaler(self):
        model = StandardScaler()
        data = numpy.array(
            [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]], dtype=numpy.float32
        )
        model.fit(data)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx, basename="CmlStandardScalerFloat32"
        )


if __name__ == "__main__":
    unittest.main()
