"""
Tests CoreML TreeEnsembleClassifier converter.
"""
import coremltools
import unittest
from sklearn.ensemble import RandomForestClassifier
from onnxmltools.convert.coreml.TreeEnsembleClassifierConverter import TreeEnsembleClassifierConverter
from onnxmltools.convert.coreml.convert import convert

class TestCoreMLTreeEnsembleClassifierConverter(unittest.TestCase):

    def validate_zipmap(self, model):
        # Validate that it contains a ZipMap
        nodes = model.graph.node
        node = next((n for n in nodes if n.op_type == 'ZipMap'), None)
        self.assertIsNotNone(node)
        self.assertEqual(len(node.output), 1)
        self.assertTrue('classProbability' in node.output)

    def test_tree_ensemble_classifier(self):
        X = [[0, 1], [1, 1], [2, 0]]
        y = [1, 0, 1]
        model = RandomForestClassifier().fit(X, y)
        model_coreml = coremltools.converters.sklearn.convert(model)
        model_onnx = convert(model_coreml.get_spec())
        self.assertTrue(model_onnx is not None)
        self.validate_zipmap(model_onnx)
