"""
Tests CoreML SupportVectorClassifier converter.
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
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLSupportVectorClassifierConverter(unittest.TestCase):

    def _fit_binary_classification(self, model):
        iris = load_iris()

        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1

        model.fit(X, y)
        return model, X[47:55].astype(numpy.float32)

    def _fit_multi_classification(self, model):
        iris = load_iris()

        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model, X[47:55].astype(numpy.float32)

    def _check_model_outputs(self, model, output_names):
        outputs = model.graph.output
        output_map = {}
        for output in outputs:
            output_map[output.name] = output

        for name in output_names:
            self.assertTrue(name in output_map)

    def validate_zipmap(self, model):
        '''
        Validate that it contains a ZipMap
        '''
        nodes = model.graph.node
        node = next((n for n in nodes if n.op_type == 'ZipMap'), None)
        self.assertIsNotNone(node)
        self.assertEqual(len(node.output), 1)
        self.assertTrue('classProbability' in node.output)

    def test_support_vector_classifier_binary_no_prob(self):
        svm, X = self._fit_binary_classification(SVC(gamma=0.5))
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        # This should not have a probability output and will be a single node
        nodes = svm_onnx.graph.node
        self.assertEqual(len(nodes), 1)
        self._check_model_outputs(svm_onnx, ['classLabel'])
        dump_data_and_model(X, svm, svm_onnx, basename="CmlBinSVC-Out0",
                            allow_failure=True)

    def test_support_vector_classifier_binary_with_prob(self):
        svm, X = self._fit_binary_classification(SVC(probability=True, gamma=0.5))
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        self.validate_zipmap(svm_onnx)
        self._check_model_outputs(svm_onnx, ['classLabel', 'classProbability'])

    def test_support_vector_classifier_multiclass_no_prob(self):
        svm, X = self._fit_multi_classification(SVC(gamma=0.5))
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        nodes = svm_onnx.graph.node
        self.assertEqual(len(nodes), 1)
        self._check_model_outputs(svm_onnx, ['classLabel'])

    def test_support_vector_classifier_multiclass_with_prob(self):
        svm, X = self._fit_multi_classification(SVC(probability=True, gamma=0.5))
        svm_coreml = coremltools.converters.sklearn.convert(svm)
        svm_onnx = convert(svm_coreml.get_spec())
        self.assertTrue(svm_onnx is not None)
        self.validate_zipmap(svm_onnx)
        self._check_model_outputs(svm_onnx, ['classLabel', 'classProbability'])


if __name__ == "__main__":
    unittest.main()
