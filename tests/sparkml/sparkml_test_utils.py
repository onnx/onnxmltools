
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import ArrayType, FloatType, DoubleType
import numpy
import pickle
import os
import warnings
from onnxmltools.utils.utils_backend import compare_backend, extract_options, evaluate_condition, is_backend_enabled, \
    OnnxRuntimeAssertionError, compare_outputs, ExpectedAssertionError
from onnxmltools.utils.utils_backend_onnxruntime import _create_column


def start_spark(options):
    import os
    import sys
    import pyspark
    executable = sys.executable
    os.environ["SPARK_HOME"] = pyspark.__path__[0]
    os.environ["PYSPARK_PYTHON"] = executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = executable

    builder = SparkSession.builder.appName("pyspark-unittesting").master("local[1]")
    if options:
        for k,v in options.items():
            builder.config(k, v)
    spark = builder.getOrCreate()

    return spark


def stop_spark(spark):
    spark.sparkContext.stop()


def save_data_models(input, expected, model, onnx_model, basename="model", folder=None):
    if folder is None:
        folder = os.environ.get('ONNXTESTDUMP', 'tests_dump')
    if not os.path.exists(folder):
        os.makedirs(folder)

    paths = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    paths.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(expected, f)

    dest = os.path.join(folder, basename + ".data.pkl")
    paths.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(input, f)

    dest = os.path.join(folder, basename + ".model")
    paths.append(dest)
    model.write().overwrite().save(dest)

    dest = os.path.join(folder, basename + ".model.onnx")
    paths.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return paths


def run_onnx_model(output_names, input, onnx_model):
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_model)
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
                raise OnnxRuntimeAssertionError(
                    "Wrong number of inputs onnx {0} != original shape {1}, onnx='{2}'".format(
                        len(inp), input.shape, onnx_model))
        elif isinstance(input, list):
            try:
                array_input = numpy.array(input)
            except Exception as e:
                raise OnnxRuntimeAssertionError(
                    "Wrong number of inputs onnx {0} != original {1}, onnx='{2}'".format(
                        len(inp), len(input), onnx_model))
            shape = sum(i.shape[1] for i in inp)
            if shape == array_input.shape[1]:
                inputs = {n.name: _create_column([row[i] for row in input], n.type) for i, n in enumerate(inp)}
            else:
                raise OnnxRuntimeAssertionError(
                    "Wrong number of inputs onnx {0} != original shape {1}, onnx='{2}'*".format(
                        len(inp), array_input.shape, onnx_model))
        else:
            raise OnnxRuntimeAssertionError(
                "Wrong number of inputs onnx {0} != original {1}, onnx='{2}'".format(
                    len(inp), len(input), onnx_model))
    else:
        raise OnnxRuntimeAssertionError(
            "Dict or list is expected, not {0}".format(type(input)))

    for k in inputs:
        if isinstance(inputs[k], list):
            inputs[k] = numpy.array(inputs[k])
    output = sess.run(output_names, inputs)
    output_shapes = [_.shape for _ in sess.get_outputs()]
    return output, output_shapes


def compare_results(expected, output, decimal=5):
    tested = 0
    if isinstance(expected, list):
        if isinstance(output, list):
            if len(expected) != len(output):
                raise OnnxRuntimeAssertionError(
                    "Unexpected number of outputs: expected={0}, got={1}".format(len(expected), len(output)))
            for exp, out in zip(expected, output):
                compare_results(exp, out, decimal=decimal)
                tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Type mismatch: output type is {0}".format(type(output)))
    elif isinstance(expected, dict):
        if not isinstance(output, dict):
            raise OnnxRuntimeAssertionError("Type mismatch fo")
        for k, v in output.items():
            if k not in expected:
                continue
            msg = compare_outputs(expected[k], v, decimal=decimal)
            if msg:
                raise OnnxRuntimeAssertionError("Unexpected output '{0}': \n{2}".format(k, msg))
            tested += 1
    elif isinstance(expected, numpy.ndarray):
        if isinstance(output, list):
            if expected.shape[0] == len(output) and isinstance(output[0], dict):
                import pandas
                output = pandas.DataFrame(output)
                output = output[list(sorted(output.columns))]
                output = output.values
        if isinstance(output, (dict, list)):
            if len(output) != 1:
                ex = str(output)
                if len(ex) > 70:
                    ex = ex[:70] + "..."
                raise OnnxRuntimeAssertionError(
                    "More than one output when 1 is expected\n{0}".format(ex))
            output = output[-1]
        if not isinstance(output, numpy.ndarray):
            raise OnnxRuntimeAssertionError(
                "output must be an array not {0}".format(type(output)))
        msg = compare_outputs(expected, output, decimal=decimal)
        if isinstance(msg, ExpectedAssertionError):
            raise msg
        if msg:
            raise OnnxRuntimeAssertionError("Unexpected output\n{1}".format(msg))
        tested += 1
    else:
        from scipy.sparse.csr import csr_matrix
        if isinstance(expected, csr_matrix):
            # DictVectorizer
            one_array = numpy.array(output)
            msg = compare_outputs(expected.todense(), one_array, decimal=decimal)
            if msg:
                raise OnnxRuntimeAssertionError("Unexpected output\n{1}".format(msg))
            tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Unexpected type for expected output ({0})".format(type(expected)))
    if tested == 0:
        raise OnnxRuntimeAssertionError("No test for model")


def dump_data_and_sparkml_model(input, expected, model, onnx=None, basename="model", folder=None,
                        backend="onnxruntime", context=None,
                        allow_failure=None, verbose=False):
    """
    Saves data with pickle, saves the model with pickle and *onnx*,
    runs and saves the predictions for the given model.
    This function is used to test a backend (runtime) for *onnx*.

    :param input: any kind of test data
    :param expected: expected data that test results must equate to
    :param model: any model
    :param onnx: *onnx* model or *None* to use *onnxmltools* to convert it
        only if the model accepts one float vector
    :param basemodel: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param folder: files are written in this folder,
        it is created if it does not exist, if *folder* is None,
        it looks first in environment variable ``ONNXTESTDUMP``,
        otherwise, it is placed into ``'tests'``.
    :param backend: backend used to compare expected output and runtime output.
        Two options are currently supported: None for no test,
        `'onnxruntime'` to use module *onnxruntime*.
    :param context: used if the model contains a custom operator such
        as a custom Keras function...
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :param verbose: additional information
    :return: the created files

    Some convention for the name,
    *Bin* for a binary classifier, *Mcl* for a multiclass
    classifier, *Reg* for a regressor, *MRg* for a multi-regressor.
    The name can contain some flags. Expected outputs refer to the
    outputs computed with the original library, computed outputs
    refer to the outputs computed with a ONNX runtime.

    * ``-CannotLoad``: the model can be converted but the runtime cannot load it
    * ``-Dec3``: compares expected and computed outputs up to 3 decimals (5 by default)
    * ``-Dec4``: compares expected and computed outputs up to 4 decimals (5 by default)
    * ``-NoProb``: The original models computed probabilites for two classes *size=(N, 2)*
      but the runtime produces a vector of size *N*, the test will compare the second column
      to the column
    * ``-OneOff``: the ONNX runtime cannot computed the prediction for several inputs,
      it must be called for each of them
      and computed output.
    * ``-Out0``: only compares the first output on both sides
    * ``-Reshape``: merges all outputs into one single vector and resizes it before comparing
    * ``-SkipDim1``: before comparing expected and computed output,
      arrays with a shape like *(2, 1, 2)* becomes *(2, 2)*

    If the *backend* is not None, the function either raises an exception
    if the comparison between the expected outputs and the backend outputs
    fails or it saves the backend output and adds it to the results.
    """
    runtime_test = dict(model=model, data=input)

    if folder is None:
        folder = os.environ.get('ONNXTESTDUMP', 'tests_dump')
    if not os.path.exists(folder):
        os.makedirs(folder)

    runtime_test['expected'] = expected

    names = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(expected, f)

    dest = os.path.join(folder, basename + ".data.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(input, f)

    dest = os.path.join(folder, basename + ".model")
    names.append(dest)
    model.write().overwrite().save(dest)

    dest = os.path.join(folder, basename + ".model.onnx")
    names.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx.SerializeToString())

    runtime_test["onnx"] = dest

    # backend
    if backend is not None:
        if not isinstance(backend, list):
            backend = [backend]
        for b in backend:
            if not is_backend_enabled(b):
                continue
            if isinstance(allow_failure, str):
                allow = evaluate_condition(b, allow_failure)
            else:
                allow = allow_failure
            if allow is None:
                output = compare_backend(b, runtime_test, options=extract_options(basename),
                                         context=context, verbose=verbose)
            else:
                try:
                    output = compare_backend(b, runtime_test, options=extract_options(basename),
                                             context=context, verbose=verbose)
                except AssertionError as e:
                    if isinstance(allow, bool) and allow:
                        warnings.warn("Issue with '{0}' due to {1}".format(basename, e))
                        continue
                    else:
                        raise e
            if output is not None:
                dest = os.path.join(folder, basename + ".backend.{0}.pkl".format(b))
                names.append(dest)
                with open(dest, "wb") as f:
                    pickle.dump(output, f)

    return names


def dataframe_to_nparray(df):
    from pyspark.ml.linalg import VectorUDT
    schema = df.schema
    npcols = []
    for i in range(0, len(df.columns)):
        if isinstance(schema.fields[i].dataType, VectorUDT):
            npcols.append(df.select(df.columns[i]).toPandas().apply(
                lambda x : numpy.array(x[0].toArray())).as_matrix().reshape(-1, 1))
        else:
            npcols.append(df.select(df.columns[i]).collect())
    return numpy.array(npcols)
