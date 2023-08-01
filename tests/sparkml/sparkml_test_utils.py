# SPDX-License-Identifier: Apache-2.0

import pickle
import os
import sys
import numpy
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument, Fail
import pyspark
from pyspark.sql import SparkSession
from onnxmltools.utils.utils_backend import (
    compare_backend,
    extract_options,
    is_backend_enabled,
    OnnxRuntimeAssertionError,
    compare_outputs,
    ExpectedAssertionError,
)


def start_spark(options):
    executable = sys.executable
    os.environ["SPARK_HOME"] = pyspark.__path__[0]
    os.environ["PYSPARK_PYTHON"] = executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = executable

    builder = SparkSession.builder.appName("pyspark-unittesting").master("local[1]")
    if options:
        for k, v in options.items():
            builder.config(k, v)
    spark = builder.getOrCreate()
    # spark.sparkContext.setLogLevel("ALL")
    return spark


def stop_spark(spark):
    spark.sparkContext.stop()


def save_data_models(
    input,
    expected,
    model,
    onnx_model,
    basename="model",
    folder=None,
    save_spark_model=False,
    pickle_spark_model=False,
    pickle_data=False,
):
    if folder is None:
        folder = os.environ.get("ONNXTESTDUMP", "tests_dump")
    if not os.path.exists(folder):
        os.makedirs(folder)

    paths = []

    if pickle_spark_model:
        dest = os.path.join(folder, basename + ".expected.pkl")
        paths.append(dest)
        with open(dest, "wb") as f:
            pickle.dump(expected, f)

    if pickle_data:
        dest = os.path.join(folder, basename + ".data.pkl")
        paths.append(dest)
        with open(dest, "wb") as f:
            pickle.dump(input, f)

    if save_spark_model:
        dest = os.path.join(folder, basename + ".model")
        paths.append(dest)
        model.write().overwrite().save(dest)

    dest = os.path.join(folder, basename + ".model.onnx")
    paths.append(dest)
    with open(dest, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return paths


def run_onnx_model(output_names, input, onnx_model):
    sess = onnxruntime.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
    if isinstance(input, dict):
        inputs = input
    elif isinstance(input, list):
        inp = sess.get_inputs()
        inputs = {i.name: v for i, v in zip(inp, input)}
    elif isinstance(input, numpy.ndarray):
        inp = sess.get_inputs()
        if len(inp) == 1:
            inputs = {inp[0].name: input}
        else:
            raise OnnxRuntimeAssertionError(
                "Wrong number of inputs onnx {0} != original shape "
                "{1}, onnx='{2}'".format(len(inp), input.shape, onnx_model)
            )
    else:
        raise OnnxRuntimeAssertionError(
            "Dict or list is expected, not {0}".format(type(input))
        )

    for k in inputs:
        if isinstance(inputs[k], list):
            inputs[k] = numpy.array(inputs[k])
    try:
        output = sess.run(output_names, inputs)
    except (InvalidArgument, Fail) as e:
        rows = []
        for inp in sess.get_inputs():
            rows.append("input: {} - {} - {}".format(inp.name, inp.type, inp.shape))
        for inp in sess.get_outputs():
            rows.append("output: {} - {} - {}".format(inp.name, inp.type, inp.shape))
        rows.append("REQUIRED: {}".format(output_names))
        for k, v in sorted(inputs.items()):
            if hasattr(v, "shape"):
                rows.append("{}={}-{}-{}".format(k, v.shape, v.dtype, v))
            else:
                rows.append("{}={}".format(k, v))
        raise AssertionError(
            "Unable to run onnxruntime\n{}".format("\n".join(rows))
        ) from e

    output_shapes = [_.shape for _ in sess.get_outputs()]
    return output, output_shapes


def compare_results(expected, output, decimal=5):
    tested = 0
    if isinstance(expected, list):
        if isinstance(output, list):
            if len(expected) != len(output):
                raise OnnxRuntimeAssertionError(
                    "Unexpected number of outputs: expected={0}, got={1}".format(
                        len(expected), len(output)
                    )
                )
            for exp, out in zip(expected, output):
                compare_results(exp, out, decimal=decimal)
                tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Type mismatch: output type is {0}".format(type(output))
            )
    elif isinstance(expected, dict):
        if not isinstance(output, dict):
            raise OnnxRuntimeAssertionError("Type mismatch fo")
        for k, v in output.items():
            if k not in expected:
                continue
            msg = compare_outputs(expected[k], v, decimal=decimal)
            if msg:
                raise OnnxRuntimeAssertionError(
                    "Unexpected output '{0}': \n{1}".format(k, msg)
                )
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
                    "More than one output when 1 is expected\n{0}".format(ex)
                )
            output = output[-1]
        if not isinstance(output, numpy.ndarray):
            raise OnnxRuntimeAssertionError(
                "output must be an array not {0}".format(type(output))
            )
        msg = compare_outputs(expected, output, decimal=decimal)
        if isinstance(msg, ExpectedAssertionError):
            raise msg
        if msg:
            raise OnnxRuntimeAssertionError("Unexpected output\n{}".format(msg))
        tested += 1
    else:
        from scipy.sparse.csr import csr_matrix

        if isinstance(expected, csr_matrix):
            # DictVectorizer
            one_array = numpy.array(output)
            msg = compare_outputs(expected.todense(), one_array, decimal=decimal)
            if msg:
                raise OnnxRuntimeAssertionError("Unexpected output\n{0}".format(msg))
            tested += 1
        else:
            raise OnnxRuntimeAssertionError(
                "Unexpected type for expected output ({0})".format(type(expected))
            )
    if tested == 0:
        raise OnnxRuntimeAssertionError("No test for model")


def dump_data_and_sparkml_model(
    input,
    expected,
    model,
    onnx=None,
    basename="model",
    folder=None,
    backend="onnxruntime",
    context=None,
    allow_failure=None,
    verbose=False,
):
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
        as a custom function...
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"pv.Version(onnx.__version__) < pv.Version('1.3.0')"``
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
    * ``-NoProb``: The original models computed probabilites
      for two classes *size=(N, 2)*
      but the runtime produces a vector of size *N*,
      the test will compare the second column
      to the column
    * ``-OneOff``: the ONNX runtime cannot computed the prediction for several inputs,
      it must be called for each of them
      and computed output.
    * ``-Out0``: only compares the first output on both sides
    * ``-Reshape``: merges all outputs into one single vector
      and resizes it before comparing
    * ``-SkipDim1``: before comparing expected and computed output,
      arrays with a shape like *(2, 1, 2)* becomes *(2, 2)*

    If the *backend* is not None, the function either raises an exception
    if the comparison between the expected outputs and the backend outputs
    fails or it saves the backend output and adds it to the results.
    """
    runtime_test = dict(model=model, data=input)

    if folder is None:
        folder = os.environ.get("ONNXTESTDUMP", "tests_dump")
    if not os.path.exists(folder):
        os.makedirs(folder)

    runtime_test["expected"] = expected

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
                raise NotImplementedError("allow_failure is deprecated.")
            output = compare_backend(
                b,
                runtime_test,
                options=extract_options(basename),
                context=context,
                verbose=verbose,
            )
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
            npcols.append(
                df.select(df.columns[i])
                .toPandas()
                .apply(lambda x: numpy.array(x[0].toArray()))
                .as_matrix()
                .reshape(-1, 1)
            )
        else:
            npcols.append(df.select(df.columns[i]).collect())
    return numpy.array(npcols)
