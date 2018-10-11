"""
Tests onnx conversion with onnxruntime.
"""
import os
import unittest
import warnings
import numpy
try:
    from .utils_backend import compare, search_converted_models, load_data_and_model, extract_options
except ImportError: 
    from utils_backend import compare, search_converted_models, load_data_and_model, extract_options
import onnxruntime


class TestBackendWithOnnxRuntime(unittest.TestCase):

    def test_onnxruntime(self):
        "Main test"
        alltests = search_converted_models()
        assert len(alltests) >= 1
        failures = []
        for test in alltests:
            if not isinstance(test, dict):
                raise TypeError("Unexpected type '{0}'".format(type(test)))
            name = os.path.split(test["onnx"])[-1].split('.')[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ImportWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                warnings.simplefilter("ignore", PendingDeprecationWarning)                
                try:
                    self._compare_model(test)
                    print("RT-OK   {}".format(name))
                except Exception as e:
                    if "DictVectorizer" in name:
                        msg = "RT-WARN {} - No suitable kernel definition found for op DictVectorizer (node DictVectorizer) - {}"
                        print(msg.format(name, str(e).replace("\n", " ").replace("\r", "")))
                    else:
                        print("RT-FAIL {} - {}".format(name, str(e).replace("\n", " ").replace("\r", "")))
                        failures.append((name, e))
        if len(failures) > 0:
            raise failures[0][1]
    
    def _compare_model(self, test, decimal=5):
        load = load_data_and_model(test)
        onnx = test['onnx']
        options = extract_options(onnx)
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
        
        OneOff = options.pop('OneOff', False)
        if OneOff:
            if len(inputs) != 1:
                raise NotImplementedError("OneOff option is not available for more than one input")            
            name, values = list(inputs.items())[0]
            res = []
            for input in values:
                try:
                    one = sess.run(None, {name: input})
                except Exception as e:
                    raise Exception("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
                res.append(one)
            output = res
        else:
            try:
                output = sess.run(None, inputs)
            except Exception as e:
                raise Exception("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
        
        self._compare_expected(load["expected"], output, sess, onnx, decimal=decimal, **options)
        
    def _compare_expected(self, expected, output, sess, onnx, decimal=5, **kwargs):
        tested = 0
        if isinstance(expected, dict):
            if not isinstance(output, dict):
                raise TypeError("Type mismatch for '{0}'".format(onnx))                
            for k, v in output.items():
                if k not in expected:
                    continue
                msg = compare(expected[k], v, decimal=decimal, **kwargs)
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
            msg = compare(expected, output, decimal=decimal, **kwargs)
            if msg:
                raise ValueError("Unexpected output in model '{0}'\n{1}".format(onnx, msg))
            tested += 1
        else:
            from scipy.sparse.csr import csr_matrix
            if isinstance(expected, csr_matrix):
                # DictVectorizer
                one_array = numpy.array(output)
                msg = compare(expected.todense(), one_array, decimal=decimal, **kwargs)
                if msg:
                    raise ValueError("Unexpected output in model '{0}'\n{1}".format(onnx, msg))
                tested += 1
            else:
                raise TypeError("Unexpected type for expected output ({1}) and onnx '{0}'".format(onnx, type(expected)))
        if tested ==0:
            raise RuntimeError("No test for onnx '{0}'".format(onnx))


if __name__ == "__main__":
    unittest.main()

