"""
Tests onnx conversion with onnxruntime.
"""
import unittest
import numpy
try:
    from .utils_backend import compare, search_converted_models, load_data_and_model
except ImportError: 
    from utils_backend import compare, search_converted_models, load_data_and_model
import onnxruntime


class TestBackendWithOnnxRuntime(unittest.TestCase):

    def test_onnxruntime(self):
        alltests = search_converted_models()
        assert len(alltests) >= 1
        for test in alltests:
            if not isinstance(test, dict):
                raise TypeError("Unexpected type '{0}'".format(type(test)))
            self.compare_model(test)
    
    def compare_model(self, test, decimal=5):
        load = load_data_and_model(test)
        onnx = test['onnx']
        try:
            sess = onnxruntime.InferenceSession(onnx)
        except Exception as e:
            raise Exception("Unable to load onnx '{0}'".format(onnx)) from e
        
        input = load["data"]
        if isinstance(input, dict):
            inputs = input
        elif isinstance(input, (list, numpy.ndarray)):
            inp = sess.get_inputs()
            if len(inp) == len(input):
                inputs = {i.name: v for i, v in zip(inp, input)}
            elif len(inp) == 1:
                inputs = {inp[0].name: input}
            else:
                raise ValueError("Wrong number of inputs {0} != {1}, onnx='{2}'".format(len(inp), len(input), onnx))
        else:
            raise TypeError("Dict or list is expected, not {0}".format(type(input)))
            
        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = numpy.array(inputs[k])
        
        try:
            output = sess.run(None, inputs)
        except Exception as e:
            raise Exception("Unable to run onnx '{0}'".format(onnx)) from e
        
        self.compare_expected(load["expected"], output, sess, onnx, decimal=decimal)
        
    def compare_expected(self, expected, output, sess, onnx, decimal=5):
        tested = 0
        if isinstance(expected, dict):
            if not isinstance(output, dict):
                raise TypeError("Type mismatch for '{0}'".format(onnx))                
            for k, v in output.items():
                if k not in expected:
                    continue
                msg = compare(expected[k], v, decimal=decimal)
                if msg:
                    raise ValueError("Unexpected output '{0}' in model '{1}'\n{2}".format(k, onnx, msg))
                tested += 1
        elif isinstance(expected, numpy.ndarray):
            if isinstance(output, (dict, list)):
                if len(output) != 1:
                    raise ValueError("More than one output when 1 is expected for onnx '{0}'".format(onnx))
                output = output.pop()
            if not isinstance(output, numpy.ndarray):
                raise TypeError("output must be an array for onnx '{0}' not {1}".format(onnx, type(output)))
            msg = compare(expected, output, decimal=decimal)
            if msg:
                raise ValueError("Unexpected output in model '{0}'\n{1}".format(onnx, msg))
            tested += 1
        else:
            raise TypeError("Unexpected type for expected output ({1}) and onnx '{0}'".format(onnx, type(expected)))
        if tested ==0:
            raise RuntimeError("No test for onnx '{0}'".format(onnx))


if __name__ == "__main__":
    unittest.main()

