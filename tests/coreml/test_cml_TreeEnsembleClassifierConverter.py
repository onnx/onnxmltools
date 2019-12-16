"""
Tests CoreML TreeEnsembleClassifier converter.
"""
import unittest
import numpy
try:
    from sklearn.impute import SimpleImputer as Imputer
    import sklearn.preprocessing
    if not hasattr(sklearn.preprocessing, 'Imputer'):
        # coremltools 3.1 does not work with scikit-learn 0.22
        setattr(sklearn.preprocessing, 'Imputer', Imputer)
except ImportError:
    from sklearn.preprocessing import Imputer
import coremltools
from sklearn.ensemble import RandomForestClassifier
from onnxmltools.convert.coreml.convert import convert
from onnxmltools.utils import dump_data_and_model


class TestCoreMLTreeEnsembleClassifierConverter(unittest.TestCase):

    def validate_zipmap(self, model):
        # Validate that it contains a ZipMap
        nodes = model.graph.node
        node = next((n for n in nodes if n.op_type == 'ZipMap'), None)
        self.assertIsNotNone(node)
        self.assertEqual(len(node.output), 1)
        self.assertTrue('classProbability' in node.output)

    def test_tree_ensemble_classifier(self):
        X = numpy.array([[0, 1], [1, 1], [2, 0]], dtype=numpy.float32)
        y = [1, 0, 1]
        model = RandomForestClassifier().fit(X, y)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        self.validate_zipmap(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="CmlBinRandomForestClassifier",
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.3.0')")


if __name__ == "__main__":
    unittest.main()
