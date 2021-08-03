# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML Scaler converter.
"""
import unittest
from distutils.version import StrictVersion
import sys
import onnx
import numpy
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from sklearn.preprocessing import StandardScaler
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLScalerConverter(unittest.TestCase):

    @unittest.skipIf(
        sys.platform == "win32" and
            StrictVersion(coremltools.__version__) <= StrictVersion("3.1") and
            StrictVersion(onnx.__version__) >= StrictVersion("1.9.0"),
        reason="incompabilities scikit-learn, coremltools")
    def test_scaler(self):
        model = StandardScaler()
        data = numpy.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]], dtype=numpy.float32)
        model.fit(data)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="CmlStandardScalerFloat32")


if __name__ == "__main__":
    unittest.main()
