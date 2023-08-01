# SPDX-License-Identifier: Apache-2.0

"""
Tests CoreML Imputer converter.
"""
import unittest
import packaging.version as pv
import numpy as np

try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing

    if not hasattr(sklearn.preprocessing, "Imputer"):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, "Imputer", Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import sklearn.preprocessing
import coremltools
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools.utils import dump_data_and_model


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestCoreMLImputerConverter(unittest.TestCase):
    @unittest.skipIf(
        pv.Version(coremltools.__version__) > pv.Version("3.1"), reason="untested"
    )
    def test_imputer(self):
        try:
            model = Imputer(missing_values="NaN", strategy="mean", axis=0)
        except TypeError:
            model = Imputer(missing_values=np.nan, strategy="mean")
            model.axis = 0
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)
        from onnxmltools.convert.coreml.convert import convert
        import coremltools  # noqa

        try:
            model_coreml = coremltools.converters.sklearn.convert(model)
        except ValueError as e:
            if "not supported" in str(e):
                # Python 2.7 + scikit-learn 0.22
                return
        model_onnx = convert(model_coreml.get_spec(), target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            np.array(data, dtype=np.float32),
            model,
            model_onnx,
            basename="CmlImputerMeanFloat32",
        )


if __name__ == "__main__":
    unittest.main()
