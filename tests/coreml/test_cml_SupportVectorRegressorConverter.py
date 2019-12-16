"""
Tests SupportVectorRegressor converter.
"""
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
import numpy
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLSupportVectorRegressorConverter(unittest.TestCase):

    def test_support_vector_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)
     
        svm = SVR(gamma=1./len(X))
        svm.fit(X, y)
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), svm, svm_onnx, basename="CmlRegSVR-Dec3")


if __name__ == "__main__":
    unittest.main()
