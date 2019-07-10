"""
Tests CoreML Imputer converter.
"""
import unittest
import numpy as np
try:
    # scikit-learn >= 0.21
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    # scikit-learn < 0.21
    from sklearn.preprocessing import Imputer
import coremltools
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLImputerConverter(unittest.TestCase):

    def test_imputer(self):
        model = Imputer(missing_values='NaN', strategy='mean')
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(np.array(data, dtype=np.float32),
                            model, model_onnx, basename="CmlImputerMeanFloat32")


if __name__ == "__main__":
    unittest.main()
