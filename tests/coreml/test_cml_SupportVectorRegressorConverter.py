"""
Tests SupportVectorRegressor converter.
"""
import coremltools
import unittest
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLSupportVectorRegressorConverter(unittest.TestCase):

    def test_support_vector_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)
     
        svm = SVR(gamma=1./len(X))
        svm.fit(X, y)
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
