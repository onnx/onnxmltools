import unittest
from onnxmltools import convert_sklearn
from onnxmltools.convert.common._data_types import FloatTensorType, Int64TensorType, StringTensorType
from onnxmltools.convert.sklearn.operator_converters.common import concatenate_variables

class TestSklearnPipeline(unittest.TestCase):

    def test_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        scaler = StandardScaler()
        scaler.fit([[0, 0],[0, 0],[1, 1],[1, 1]])
        model = Pipeline([('scaler1',scaler),('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline', [FloatTensorType([1, 2])])
        self.assertTrue(model_onnx is not None)

    def test_combine_inputs(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        scaler = StandardScaler()
        scaler.fit([[0., 0.],[0., 0.],[1., 1.],[1., 1.]])
        model = Pipeline([('scaler1', scaler),('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline', [FloatTensorType([1, 1]), FloatTensorType([1, 1])])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)

    def test_combine_inputs_floats_ints(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        scaler = StandardScaler()
        scaler.fit([[0, 0.],[0, 0.],[1, 1.],[1, 1.]])
        model = Pipeline([('scaler1', scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline', [Int64TensorType([1, 1]), FloatTensorType([1, 1])])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)

    def test_combine_inputs_with_string(self):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import make_pipeline

        model = LabelEncoder()
        model.fit(['a', 'b', 'b', 'a', 'c'])

        model_onnx = convert_sklearn(model, 'pipeline', [StringTensorType([1, 1]), StringTensorType([1, 4])])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)

