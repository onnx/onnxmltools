import unittest
import numpy
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from onnxmltools.utils import dump_data_and_model


class PipeConcatenateInput:
    def __init__(self, pipe):
        self.pipe = pipe
        
    def transform(self, inp):
        if isinstance(inp, numpy.ndarray):
            return self.pipe.transform(inp)
        elif isinstance(inp, dict):
            keys = list(sorted(inp.keys()))
            dim = inp[keys[0]].shape[0], len(keys)
            x2 = numpy.zeros(dim)
            for i in range(x2.shape[1]):
                x2[:, i] = inp[keys[i]]
            res = self.pipe.transform(x2)
            return res
        else:
            raise TypeError("Unable to predict with type {0}".format(type(X)))


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
        data = {'input1': data[:, 0], 'input2': data[:, 1]}
        dump_data_and_model(data, PipeConcatenateInput(model), model_onnx,
                            basename="SklearnPipelineScaler11-OneOff")

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
        data = numpy.array(data)
        data = {'input1': data[:, 0].astype(numpy.int64), 
                'input2': data[:, 1].astype(numpy.float32)}
        dump_data_and_model(data, PipeConcatenateInput(model), model_onnx,
                            basename="SklearnPipelineScalerMixed-OneOff")


if __name__ == "__main__":
    unittest.main()
