# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML GLMRegressor converter.
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
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestCoreMLGLMRegressorConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(coremltools.__version__) > pv.Version("3.1"), reason="untested"
    )
    def test_glm_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)

        lr = LinearRegression()
        lr.fit(X, y)
        lr_coreml = coremltools.converters.sklearn.convert(lr)
        lr_onnx = convert(lr_coreml.get_spec(), target_opset=TARGET_OPSET)
        self.assertTrue(lr_onnx is not None)
        dump_data_and_model(
            X.astype(numpy.float32), lr, lr_onnx, basename="CmlLinearRegression-Dec4"
        )

        svr = LinearSVR()
        svr.fit(X, y)
        svr_coreml = coremltools.converters.sklearn.convert(svr)
        svr_onnx = convert(svr_coreml.get_spec(), target_opset=TARGET_OPSET)
        self.assertTrue(svr_onnx is not None)
        dump_data_and_model(
            X.astype(numpy.float32), svr, svr_onnx, basename="CmlLinearSvr-Dec4"
        )


if __name__ == "__main__":
    unittest.main()
