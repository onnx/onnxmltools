"""
Tests CoreML Imputer converter.
"""
import numpy as np
import unittest
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import sklearn.preprocessing
from onnxmltools.utils import dump_data_and_model


class TestCoreMLImputerConverter(unittest.TestCase):

    def test_imputer(self):
        try:
            model = Imputer(missing_values='NaN', strategy='mean', axis=0)
        except TypeError:
            model = Imputer(missing_values=np.nan, strategy='mean')
        data = [[1, 2], [np.nan, 3], [7, 6]]
        model.fit(data)
        from onnxmltools.convert.coreml.convert import convert
        import coremltools  # noqa
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(np.array(data, dtype=np.float32),
                            model, model_onnx, basename="CmlImputerMeanFloat32")


if __name__ == "__main__":
    unittest.main()
