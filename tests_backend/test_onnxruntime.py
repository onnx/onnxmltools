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
    from .utils_backend import compare, search_converted_models, load_data_and_model, extract_options, ExpectedAssertionError, OnnxRuntimeAssertionError
except ImportError: 
    from utils_backend import compare, search_converted_models, load_data_and_model, extract_options, ExpectedAssertionError, OnnxRuntimeAssertionError
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
                raise OnnxRuntimeAssertionError("Unexpected type '{0}'".format(type(test)))
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
                    elif isinstance(e, ExpectedAssertionError):
                        msg = "RT-WARN {} - {}".format(name, str(e).replace("\n", " ").replace("\r", ""))
                    else:
                        msg = "RT-FAIL {} - {}".format(name, str(e).replace("\n", " ").replace("\r", ""))
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

    def _create_column(self, values, dtype):
        "Creates a column from values with dtype"
        if str(dtype) == "tensor(int64)":
            return numpy.array(values, dtype=numpy.int64)
        elif str(dtype) == "tensor(float)":
            return numpy.array(values, dtype=numpy.float32)
        else:
            raise OnnxRuntimeAssertionError("Unable to create one column from dtype '{0}'".format(dtype))

    def _compare_model(self, test, decimal=5, verbose=False):
        load = load_data_and_model(test)
        onnx = test['onnx']
        options = extract_options(onnx)
        try:
            sess = onnxruntime.InferenceSession(onnx)
        except ExpectedAssertionError as expe:
            raise expe
        except Exception as e:
            if "-CannotLoad" in onnx:
                raise ExpectedAssertionError("Unable to load onnx '{0}' due to\n{1}".format(onnx, e)) from e
            else:
                raise OnnxRuntimeAssertionError("Unable to load onnx '{0}'".format(onnx)) from e
        
        input = load["data"]
        if isinstance(input, dict):
            inputs = input
        elif isinstance(input, (list, numpy.ndarray)):
            inp = sess.get_inputs()
            if len(inp) == len(input):
                inputs = {i.name: v for i, v in zip(inp, input)}
            elif len(inp) == 1:
                inputs = {inp[0].name: input}
            elif isinstance(input, numpy.ndarray):
                shape = sum(i.shape[1] if len(i.shape) == 2 else i.shape[0] for i in inp)
                if shape == input.shape[1]:
                    inputs = {n.name: input[:, i] for i, n in enumerate(inp)}
                else:
                    raise OnnxRuntimeAssertionError("Wrong number of inputs onnx {0} != original shape {1}, onnx='{2}'".format(len(inp), input.shape, onnx))
            elif isinstance(input, list):
                try:
                    array_input = numpy.array(input)
                except Exception as e:
                    raise OnnxRuntimeAssertionError("Wrong number of inputs onnx {0} != original {1}, onnx='{2}'".format(len(inp), len(input), onnx))
                shape = sum(i.shape[1] for i in inp)
                if shape == array_input.shape[1]:
                    inputs = {n.name: self._create_column([row[i] for row in input], n.type) for i, n in enumerate(inp)}
                else:
                    raise OnnxRuntimeAssertionError("Wrong number of inputs onnx {0} != original shape {1}, onnx='{2}'*".format(len(inp), array_input.shape, onnx))
            else:
                raise OnnxRuntimeAssertionError("Wrong number of inputs onnx {0} != original {1}, onnx='{2}'".format(len(inp), len(input), onnx))
        else:
            raise OnnxRuntimeAssertionError("Dict or list is expected, not {0}".format(type(input)))
            
        for k in inputs:
            if isinstance(inputs[k], list):
                inputs[k] = numpy.array(inputs[k])
        
        OneOff = options.pop('OneOff', False)
        if OneOff:
            if len(inputs) == 1:
                name, values = list(inputs.items())[0]
                res = []
                for input in values:
                    try:
                        one = sess.run(None, {name: input})
                    except ExpectedAssertionError as expe:
                        raise expe
                    except Exception as e:
                        raise OnnxRuntimeAssertionError("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
                    res.append(one)
                output = self._post_process_output(res)
            else:
                t = list(inputs.items())[0]
                res = []
                for i in range(0, len(t[1])):
                    iii = {k: numpy.array([v[i]], dtype=numpy.float32) for k, v in inputs.items()}
                    try:
                        one = sess.run(None, iii)
                    except ExpectedAssertionError as expe:
                        raise expe
                    except Exception as e:
                        raise OnnxRuntimeAssertionError("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
                    res.append(one)
                output = self._post_process_output(res)                
        else:
            try:
                output = sess.run(None, inputs)
            except ExpectedAssertionError as expe:
                raise expe
            except RuntimeError as e:
                if "-Fail" in onnx:
                    raise ExpectedAssertionError("onnxruntime cannot compute the prediction for '{0}'".format(onnx)) from e
                else:
                    raise OnnxRuntimeAssertionError("onnxruntime cannot compute the prediction for '{0}'".format(onnx)) from e
            except Exception as e:
                raise OnnxRuntimeAssertionError("Unable to run onnx '{0}' due to {1}".format(onnx, e)) from e
        
        output0 = output.copy()

        try:
            self._compare_expected(load["expected"], output, sess, onnx, decimal=decimal, **options)
        except ExpectedAssertionError as expe:
            raise expe
        except Exception as e:
            raise OnnxRuntimeAssertionError("Model '{0}' has discrepencies.\n{1}: {2}".format(onnx, type(e), e)) from e
        
    def _compare_expected(self, expected, output, sess, onnx, decimal=5, **kwargs):
        tested = 0
        if isinstance(expected, list):
            if isinstance(output, list):
                if 'Out0' in kwargs:
                    expected = expected[:1]
                    output = output[:1]
                    del kwargs['Out0']
                if 'Reshape' in kwargs:
                    del kwargs['Reshape']
                    output = numpy.hstack(output).ravel()
                    output = output.reshape((len(expected),
                                             len(output.ravel()) // len(expected)))
                if len(expected) != len(output):
                    raise OnnxRuntimeAssertionError("Unexpected number of outputs '{0}', expected={1}, got={2}".format(onnx, len(expected), len(output)))
                for exp, out in zip(expected, output):
                    self._compare_expected(exp, out, sess, onnx, decimal=5, **kwargs)
                    tested += 1
            else:
                raise OnnxRuntimeAssertionError("Type mismatch for '{0}', output type is {1}".format(onnx, type(output)))
        elif isinstance(expected, dict):
            if not isinstance(output, dict):
                raise OnnxRuntimeAssertionError("Type mismatch for '{0}'".format(onnx))                
            for k, v in output.items():
                if k not in expected:
                    continue
                msg = compare(expected[k], v, decimal=decimal, **kwargs)
                if msg:
                    raise OnnxRuntimeAssertionError("Unexpected output '{0}' in model '{1}'\n{2}".format(k, onnx, msg)) from msg
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
                    raise OnnxRuntimeAssertionError("More than one output when 1 is expected for onnx '{0}'\n{1}".format(onnx, ex))
                output = output[-1]
            if not isinstance(output, numpy.ndarray):
                raise OnnxRuntimeAssertionError("output must be an array for onnx '{0}' not {1}".format(onnx, type(output)))
            msg = compare(expected, output, decimal=decimal, **kwargs)
            if isinstance(msg, ExpectedAssertionError):
                raise msg
            if msg:
                raise OnnxRuntimeAssertionError("Unexpected output in model '{0}'\n{1}".format(onnx, msg)) from msg
            tested += 1
        else:
            from scipy.sparse.csr import csr_matrix
            if isinstance(expected, csr_matrix):
                # DictVectorizer
                one_array = numpy.array(output)
                msg = compare(expected.todense(), one_array, decimal=decimal, **kwargs)
                if msg:
                    raise OnnxRuntimeAssertionError("Unexpected output in model '{0}'\n{1}".format(onnx, msg)) from msg
                tested += 1
            else:
                raise OnnxRuntimeAssertionError("Unexpected type for expected output ({1}) and onnx '{0}'".format(onnx, type(expected)))
        if tested ==0:
            raise OnnxRuntimeAssertionError("No test for onnx '{0}'".format(onnx))


if __name__ == "__main__":
    unittest.main()

