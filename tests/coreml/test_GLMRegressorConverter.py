"""
Tests CoreML GLMRegressor converter.
"""
import coremltools
import unittest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLGLMRegressorConverter(unittest.TestCase):

    def test_glm_regressor(self):
        X, y = make_regression(n_features=4, random_state=0)

        lr = LinearRegression()
        lr.fit(X, y)
        lr_coreml = coremltools.converters.sklearn.convert(lr)
        lr_onnx = convert(lr_coreml.get_spec())
        self.assertTrue(lr_onnx is not None)

        svr = LinearSVR()
        svr.fit(X, y)
        svr_coreml = coremltools.converters.sklearn.convert(svr)
        svr_onnx = convert(svr_coreml.get_spec())
        self.assertTrue(svr_onnx is not None)