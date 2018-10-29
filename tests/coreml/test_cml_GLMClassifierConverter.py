"""
Tests CoreML GLMClassifier converter.
"""
import coremltools
import unittest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from onnxmltools.convert.coreml.convert import convert

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

        # Ensure there is a probability output
        svm = LinearSVC()
        svm.fit(X, y)
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        self.validate_zipmap(svm_onnx)


if __name__ == "__main__":
    unittest.main()
