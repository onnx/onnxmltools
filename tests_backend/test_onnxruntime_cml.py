"""
Tests onnx conversion with onnxruntime.
"""
import unittest
try:
    from .main_test_onnxruntime import MainTestBackendWithOnnxRuntime
except ImportError: 
    from main_test_onnxruntime import MainTestBackendWithOnnxRuntime


class TestBackendWithOnnxRuntime_Cml(MainTestBackendWithOnnxRuntime):
    
    prefix = 'Cml'
    
    @classmethod
    def setUpClass(cls):
        MainTestBackendWithOnnxRuntime._setUpClass(TestBackendWithOnnxRuntime_Cml)
    
    @classmethod
    def tearDownClass(cls):
        MainTestBackendWithOnnxRuntime._tearDownClass(TestBackendWithOnnxRuntime_Cml)
    
    def test_CmlLinearRegression_Dec4(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlLinearSvr_Dec4(self): self._main_test_onnxruntime(self._testMethodName)        
    @unittest.skip("prediction mismatch")
    def test_CmlBinLinearSVC_NoProb(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlBinRandomForestClassifier(self): self._main_test_onnxruntime(self._testMethodName)        
    @unittest.skip("prediction mismatch")
    def test_CmlBinSVC_Out0(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlDictVectorizer_OneOff_SkipDim1(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlImputerMeanFloat32(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlRegRandomForestRegressor_Dec4(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlRegSVR_Dec3(self): self._main_test_onnxruntime(self._testMethodName)    
    def test_CmlStandardScalerFloat32(self): self._main_test_onnxruntime(self._testMethodName)        
    @unittest.skip("prediction mismatch")
    def test_CmlXGBoostRegressor_OneOff_Reshape(self): self._main_test_onnxruntime(self._testMethodName)        
    def test_CmlbinLogitisticRegression(self): self._main_test_onnxruntime(self._testMethodName)
        

if __name__ == "__main__":
    unittest.main()

