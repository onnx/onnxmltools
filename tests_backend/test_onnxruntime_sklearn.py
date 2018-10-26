"""
Tests onnx conversion with onnxruntime.
"""
import unittest
try:
    from .main_test_onnxruntime import MainTestBackendWithOnnxRuntime
except ImportError: 
    from main_test_onnxruntime import MainTestBackendWithOnnxRuntime

try:
    from onnxruntime import __version__ as rt_version
    from onnxmltools.convert.common.utils import compare_strict_version
except ImportError:
    rt_version = None
    compare_strict_version = lambda a,b: 1


class TestBackendWithOnnxRuntime_Sklearn(MainTestBackendWithOnnxRuntime):
    
    prefix = 'Sklearn'
    
    @classmethod
    def setUpClass(cls):
        MainTestBackendWithOnnxRuntime._setUpClass(TestBackendWithOnnxRuntime_Sklearn)
    
    @classmethod
    def tearDownClass(cls):
        MainTestBackendWithOnnxRuntime._tearDownClass(TestBackendWithOnnxRuntime_Sklearn)    
    
    @unittest.skip("prediction mismatch")
    def test_SklearnBinBernoulliNB_OneOff(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinDecisionTreeClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinExtraTreesClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinGradientBoostingClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("prediction mismatch")
    def test_SklearnBinLGBMClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skipIf(compare_strict_version(rt_version, "0.1.3") <= 0, reason="fixed in newer versions")
    def test_SklearnBinMultinomialNB_OneOff(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("cannot compute predictions")
    def test_SklearnBinNuSVCPF(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinNuSVCPT(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinRandomForestClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("cannot compute predictions")
    def test_SklearnBinSVCLinearPF(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnBinarizer_SkipDim1(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnDictVectorizer_OneOff_SkipDim1(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnElasticNet_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnImputerMeanFloat32(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLabelEncoder(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLassoLars_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLinearRegression_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLinearSVCBinary_NoProb(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLinearSVCMulti(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLinearSvr_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLogitisticRegressionBinary(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnLogitisticRegressionMulti(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMRgDecisionTreeRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMRgExtraTreesRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMRgRandomForestRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMaxAbsScaler(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skipIf(compare_strict_version(rt_version, "0.1.3") <= 0, reason="fixed in newer versions")
    def test_SklearnMclBernoulliNB_OneOff(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMclDecisionTreeClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMclExtraTreesClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("prediction mismatch")
    def test_SklearnMclLGBMClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skipIf(compare_strict_version(rt_version, "0.1.3") <= 0, reason="fixed in newer versions")
    def test_SklearnMclMultinomialNB_OneOff(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("prediction mismatch")
    def test_SklearnMclNuSVCPF(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMclNuSVCPT(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMclRandomForestClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("prediction mismatch")
    def test_SklearnMclSVCLinearPF(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnMinMaxScaler(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnNormalizerL2_SkipDim1(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnOneDecisionTreeClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnOneExtraTreesClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnOneHotEncoderInt64_SkipDim1(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("cannot load onnx file")
    def test_SklearnOneHotEncoderStringInt64(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnOneRandomForestClassifier(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnPipelineScaler(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("dimension mismatch")
    def test_SklearnPipelineScaler11(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("dimension mismatch")
    def test_SklearnPipelineScalerMixed(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegDecisionTreeRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegExtraTreesRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegGradientBoostingRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    @unittest.skip("prediction mismatch")
    def test_SklearnRegLGBMRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegNuSVR(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegNuSVR2(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegRandomForestRegressor(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRegSVRLinear_Dec3(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRidge_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRobustScalerFloat32(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRobustScalerNoScalingFloat32(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnRobustScalerWithCenteringFloat32(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnSGDClassifierBinary_NoProb_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnSGDClassifierMulti_Dec3(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnSGDRegressor_Dec4(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnStandardScalerFloat32(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnStandardScalerInt64(self): self._main_test_onnxruntime(self._testMethodName)
    def test_SklearnTruncatedSVD(self): self._main_test_onnxruntime(self._testMethodName)


if __name__ == "__main__":
    unittest.main()

