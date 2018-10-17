"""
Tests onnx conversion with onnxruntime.
"""
import math
import os
import unittest
import warnings
import numpy
import pandas
from onnxmltools.convert.common.data_types import FloatTensorType
try:
    from .utils_backend import compare, search_converted_models, load_data_and_model, extract_options, ExpectedAssertionError
except ImportError: 
    from utils_backend import compare, search_converted_models, load_data_and_model, extract_options, ExpectedAssertionError
import onnxruntime


class TestBackendWithOnnxRuntime(unittest.TestCase):

    def test_onnxruntime(self):
        "Main test"
        alltests = search_converted_models()
        assert len(alltests) >= 1
        failures = []
        status = []
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
                    msg = "RT-OK   {}".format(name)
                except Exception as e:
                    if "DictVectorizer" in name:
                        msg = "RT-WARN {} - No suitable kernel definition found for op DictVectorizer (node DictVectorizer) - {}"
                        msg = msg.format(name, str(e).replace("\n", " ").replace("\r", ""))
                    else:
                        msg = "RT-FAIL {} - {}".format(name, str(e).replace("\n", " ").replace("\r", ""))
                        if not isinstance(e, ExpectedAssertionError):
                            failures.append((name, e))
            status.append(msg)
        # To let the status be displayed by pytest.
        warnings.warn("\n" + "\n".join(status) + "\n")
        if len(failures) > 0:
            raise failures[0][1]
            
    def _post_process_output(self, res):
        if isinstance(res, list):
            if len(res) == 0:
                return res
            elif len(res) == 1:
                return self._post_process_output(res[0])
            elif isinstance(res[0], numpy.ndarray):
                return numpy.array(res)
            elif isinstance(res[0], dict):
                return pandas.DataFrame(res).values
            else:
                ls = [len(r) for r in res]
                mi = min(ls)
                if mi != max(ls):
                    raise NotImplementedError("Unable to postprocess various number of outputs in [{0}, {1}]".format(min(ls), max(ls)))
                if mi > 1:
                    output = []
                    for i in range(mi):
                        output.append(self._post_process_output([r[i] for r in res]))
                    return output
                elif isinstance(res[0], list):
                    # list of lists
                    if isinstance(res[0][0], list):
                        return numpy.array(res)
                    elif len(res[0]) == 1 and isinstance(res[0][0], dict):
                        return self._post_process_output([r[0] for r in res])
                    else:
                        return res
                else:
                    return res
        else:
            return res
    
    def _compare_model(self, test, decimal=5, verbose=False):
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
            output = self._post_process_output(res)
        else:
            try:
                output = sess.run(None, inputs)
            except Exception as e:
                raise Exception("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
        
        try:
            self._compare_expected(load["expected"], output, sess, onnx, decimal=decimal, **options)
        except ExpectedAssertionError as expe:
            raise expe
        except Exception as e:
            raise AssertionError("Model '{0}' has discrepencies.\n{0}".format(onnx, e))
        
    def _compare_expected(self, expected, output, sess, onnx, decimal=5, **kwargs):
        tested = 0
        if isinstance(expected, list):
            if isinstance(output, list):
                if len(expected) != len(output):
                    raise ValueError("Unexpected number of outputs '{0}', expected={1}, got={2}".format(onnx, len(expected), len(output)))
                for exp, out in zip(expected, output):
                    self._compare_expected(exp, out, sess, onnx, decimal=5, **kwargs)
                    tested += 1
            else:
                raise TypeError("Type mismatch for '{0}', output type is {1}".format(onnx, type(output)))
        elif isinstance(expected, dict):
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
            if isinstance(output, list):
                if expected.shape[0] == len(output) and isinstance(output[0], dict):
                    output = pandas.DataFrame(output)
                    output = output[list(sorted(output.columns))]
                    output = output.values
            if isinstance(output, (dict, list)):
                if len(output) != 1:
                    ex = str(output)
                    if len(ex) > 70:
                        ex = ex[:70] + "..."
                    raise ValueError("More than one output when 1 is expected for onnx '{0}'\n{1}".format(onnx, ex))
                output = output.pop()
            if not isinstance(output, numpy.ndarray):
                raise TypeError("output must be an array for onnx '{0}' not {1}".format(onnx, type(output)))
            msg = compare(expected, output, decimal=decimal, **kwargs)
            if isinstance(msg, ExpectedAssertionError):
                raise msg
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

