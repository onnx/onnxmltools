"""
Tests onnx conversion with onnxruntime.
"""
import unittest
try:
    from .main_test_onnxruntime import MainTestBackendWithOnnxRuntime
except ImportError: 
    from main_test_onnxruntime import MainTestBackendWithOnnxRuntime


class TestBackendWithOnnxRuntime_LightGbm(MainTestBackendWithOnnxRuntime):
    
    prefix = 'LightGbm'
    
    @classmethod
    def setUpClass(cls):
        MainTestBackendWithOnnxRuntime._setUpClass(TestBackendWithOnnxRuntime_LightGbm)
    
    @classmethod
    def tearDownClass(cls):
        MainTestBackendWithOnnxRuntime._tearDownClass(TestBackendWithOnnxRuntime_LightGbm)    
    
    def test_LightGbmBinLGBMClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_LightGbmMclLGBMClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_LightGbmRegLGBMRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_LightGbmRegLGBMRegressor1(self): self._main_test_onnxruntime(self._testMethodName)
    def test_LightGbmRegLGBMRegressor2(self): self._main_test_onnxruntime(self._testMethodName)


if __name__ == "__main__":
    unittest.main()
