"""
Main functions to convert machine learned model from *Core ML* model to *ONNX*.
"""
import os
import unittest
import warnings
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
from sklearn.preprocessing import OneHotEncoder
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils.tests_helper import dump_data_and_model


class TestCoremlOneHotEncoderConverter(unittest.TestCase):

    def test_one_hot_encoder(self):
        script_dir = os.path.dirname(__file__)
        relative_path = "../data/onehot_simple.mlmodel"
        abs_file = os.path.join(script_dir, relative_path)
        model_coreml = coremltools.utils.load_spec(abs_file)
        model_onnx = convert(model_coreml)
        self.assertTrue(model_onnx is not None)

    @unittest.skip('broken with the dump_data_and_model change.')
    def test_conversion_one_column(self):
        scikit_data = [[0], [1], [2], [4], [3], [2], [4], [5], [6], [7]]
        scikit_data = numpy.asarray(scikit_data, dtype='d')
        scikit_data_multiple_cols = [[0, 1],  [1, 0], [2, 2], [3, 3], [4, 4]]
        scikit_data_multiple_cols = numpy.asarray(scikit_data_multiple_cols, dtype='d')
        scikit_model = OneHotEncoder()

        # scikit_model.fit(scikit_data)
        # model_coreml = coremltools.converters.sklearn.convert(scikit_model, 'single_feature', 'out')

        scikit_model.fit(scikit_data_multiple_cols)
        try:
            model_coreml = coremltools.converters.sklearn.convert(scikit_model, ['feature_1', 'feature_2'], 'out')
        except Exception as e:
            warnings.warn("Unable to run convert OneHotEncoder with coreml.")
            return
        model_onnx = convert(model_coreml)                                     
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(scikit_data, scikit_model, model_onnx, basename="CmlOneHotEncoder-SkipDim1")
            


if __name__ == "__main__":
    unittest.main()
