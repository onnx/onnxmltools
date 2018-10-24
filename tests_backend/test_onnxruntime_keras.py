"""
Tests onnx conversion with onnxruntime.
"""
import unittest
try:
    from .main_test_onnxruntime import MainTestBackendWithOnnxRuntime
except ImportError: 
    from main_test_onnxruntime import MainTestBackendWithOnnxRuntime


class TestBackendWithOnnxRuntime_Keras(MainTestBackendWithOnnxRuntime):
    
    prefix = 'Keras'
    
    @classmethod
    def setUpClass(cls):
        MainTestBackendWithOnnxRuntime._setUpClass(TestBackendWithOnnxRuntime_Keras)
    
    @classmethod
    def tearDownClass(cls):
        MainTestBackendWithOnnxRuntime._tearDownClass(TestBackendWithOnnxRuntime_Keras)    
    
    def test_KerasCustomOp_Out0(self): self._main_test_onnxruntime(self._testMethodName)


if __name__ == "__main__":
    unittest.main()

