"""
Tests CoreML GLMClassifier converter.
"""
import unittest
import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLGLMClassifierConverter(unittest.TestCase):

    def validate_zipmap(self, model):
        # Validate that it contains a ZipMap
        nodes = model.graph.node
        node = next((n for n in nodes if n.op_type == 'ZipMap'), None)
        self.assertIsNotNone(node)
        self.assertEqual(len(node.output), 1)
        self.assertTrue('classProbability' in node.output)

    def test_glm_classifier(self):
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1

        # scikit-learn has changed the default value for multi_class.
        lr = LogisticRegression(multi_class='ovr')
        lr.fit(X, y)
        lr_coreml = coremltools.converters.sklearn.convert(lr)
        lr_onnx = convert(lr_coreml.get_spec())
        self.assertTrue(lr_onnx is not None)
        self.validate_zipmap(lr_onnx)
        dump_data_and_model(X.astype(numpy.float32), lr, lr_onnx, basename="CmlbinLogitisticRegression",
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")

        # Ensure there is a probability output
        svm = LinearSVC()
        svm.fit(X, y)
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        self.validate_zipmap(svm_onnx)
        dump_data_and_model(X.astype(numpy.float32), svm, svm_onnx, basename="CmlBinLinearSVC-NoProb",
                            allow_failure=True)


if __name__ == "__main__":
    unittest.main()
