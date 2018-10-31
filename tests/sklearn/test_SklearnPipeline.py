import unittest
import numpy
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnPipeline(unittest.TestCase):

    def test_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1',scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelineScaler")

    def test_combine_inputs(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]], dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1', scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline',
                                     [('input1', FloatTensorType([1, 1])),
                                      ('input2', FloatTensorType([1, 1]))])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelineScaler11",
                            allow_failure=True)

    def test_combine_inputs_floats_ints(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = [[0, 0.],[0, 0.],[1, 1.],[1, 1.]]
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1', scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline',
                                     [('input1', Int64TensorType([1, 1])),
                                      ('input2', FloatTensorType([1, 1]))])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelineScalerMixed",
                            allow_failure=True)


if __name__ == "__main__":
    # TestSklearnTreeEnsembleModels().test_decision_tree_regressor()
    unittest.main()
