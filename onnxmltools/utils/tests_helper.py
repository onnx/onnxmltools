#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------
import numpy
import pickle
import os
import warnings
from ..convert.common.data_types import FloatTensorType
from .utils_backend import compare_backend, extract_options, evaluate_condition, is_backend_enabled


def dump_data_and_model(data, model, onnx=None, basename="model", folder=None,
                        inputs=None, backend="onnxruntime", context=None,
                        allow_failure=None, verbose=False):
    """
    Saves data with pickle, saves the model with pickle and *onnx*,
    runs and saves the predictions for the given model.
    This function is used to test a backend (runtime) for *onnx*.

    :param data: any kind of data
    :param model: any model
    :param onnx: *onnx* model or *None* to use *onnxmltools* to convert it
        only if the model accepts one float vector
    :param basemodel: three files are writen ``<basename>.data.pkl``,
        ``<basename>.model.pkl``, ``<basename>.model.onnx``
    :param folder: files are written in this folder,
        it is created if it does not exist, if *folder* is None,
        it looks first in environment variable ``ONNXTESTDUMP``,
        otherwise, it is placed into ``'tests'``.
    :param inputs: standard type or specific one if specified, only used is
        parameter *onnx* is None
    :param backend: backend used to compare expected output and runtime output.
        Two options are currently supported: None for no test,
        `'onnxruntime'` to use module *onnxruntime*.
    :param context: used if the model contains a custom operator
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :param verbose: prints more information when it fails
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
    runtime_test = dict(model=model, data=data)

    if folder is None:
        folder = os.environ.get('ONNXTESTDUMP', 'tests')
    if not os.path.exists(folder):
        os.makedirs(folder)

    if hasattr(model, "predict"):
        import lightgbm
        if isinstance(model, lightgbm.Booster):
            # LightGBM Booster
            model_dict = model.dump_model()
            if model_dict['objective'].startswith('binary'):
                score = model.predict(data)
                prediction = [score > 0.5, numpy.vstack([1-score, score]).T]
            elif model_dict['objective'].startswith('multiclass'):
                score = model.predict(data)
                prediction = [score.argmax(axis=1), score]
            else:
                prediction = [model.predict(data)]
        elif hasattr(model, "predict_proba"):
            # Classifier
            prediction = [model.predict(data), model.predict_proba(data)]
        elif hasattr(model, "decision_function"):
            # Classifier without probabilities
            prediction = [model.predict(data), model.decision_function(data)]
        elif hasattr(model, "layers"):
            # Keras
            if len(model.input_names) != 1:
                raise NotImplemented("Only neural network with one input are supported")
            prediction = [model.predict(data)]
        else:
            # Regressor
            prediction = [model.predict(data)]
    elif hasattr(model, "transform"):
        prediction = model.transform(data)
    else:
        raise TypeError("Model has not predict or transform method: {0}".format(type(model)))

    runtime_test['expected'] = prediction

    names = []
    dest = os.path.join(folder, basename + ".expected.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(prediction, f)

    dest = os.path.join(folder, basename + ".data.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(data, f)

    dest = os.path.join(folder, basename + ".model.pkl")
    names.append(dest)
    with open(dest, "wb") as f:
        pickle.dump(model, f)

    if onnx is None:
        array = numpy.array(data)
        if inputs is None:
            inputs = [('input', FloatTensorType(list(array.shape)))]
        onnx, _ = convert_model(model, basename, inputs)

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


def convert_model(model, name, input_types):
    """
    Runs the appropriate conversion method.

    :param model: model
    :return: *onnx* model
    """
    from sklearn.base import BaseEstimator
    if model.__class__.__name__.startswith("LGBM"):
        from onnxmltools.convert import convert_lightgbm
        model, prefix = convert_lightgbm(model, name, input_types), "LightGbm"
    elif model.__class__.__name__.startswith("XGB"):
        from onnxmltools.convert import convert_xgboost
        model, prefix = convert_xgboost(model, name, input_types), "XGB"
    elif model.__class__.__name__ == 'Booster':
        import lightgbm
        if isinstance(model, lightgbm.Booster):
            from onnxmltools.convert import convert_lightgbm
            model, prefix = convert_lightgbm(model, name, input_types), "LightGbm"
        else:
            raise RuntimeError("Unable to convert model of type '{0}'.".format(type(model)))
    elif isinstance(model, BaseEstimator):
        from onnxmltools.convert import convert_sklearn
        model, prefix = convert_sklearn(model, name, input_types), "Sklearn"
    else:
        from onnxmltools.convert import convert_coreml
        model, prefix = convert_coreml(model, name, input_types), "Cml"
    if model is None:
        raise RuntimeError("Unable to convert model of type '{0}'.".format(type(model)))
    return model, prefix


def dump_one_class_classification(model, suffix="", folder=None, allow_failure=None):
    """
    Trains and dumps a model for a One Class outlier problem.
    The function trains a model and calls
    :func:`dump_data_and_model`.

    :param model: any model following *scikit-learn* API
    :param suffix: added to filenames
    :param folder: where to save the file
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :return: output of :func:`dump_data_and_model`

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0., 1.], [1., 1.], [2., 0.]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [1, 1, 1]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, 'one_class', [('input', FloatTensorType([1, 2]))])
    return dump_data_and_model(X, model, model_onnx, folder=folder, allow_failure=allow_failure,
                               basename=prefix + "One" + model.__class__.__name__ + suffix)


def dump_binary_classification(model, suffix="", folder=None, allow_failure=None, verbose=False):
    """
    Trains and dumps a model for a binary classification problem.

    :param model: any model following *scikit-learn* API
    :param suffix: added to filenames
    :param folder: where to save the file
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :param verbose: prints more information when it fails
    :return: output of :func:`dump_data_and_model`

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [0, 1, 0]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, 'tree-based binary classifier', [('input', FloatTensorType([1, 2]))])
    dump_data_and_model(X, model, model_onnx, folder=folder, allow_failure=allow_failure,
                        basename=prefix + "Bin" + model.__class__.__name__ + suffix,
                        verbose=verbose)

def dump_multiple_classification(model, suffix="", folder=None, allow_failure=None):
    """
    Trains and dumps a model for a binary classification problem.

    :param model: any model following *scikit-learn* API
    :param suffix: added to filenames
    :param folder: where to save the file
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :return: output of :func:`dump_data_and_model`

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
    X = numpy.array(X, dtype=numpy.float32)
    y = [0, 1, 2, 1, 1, 2]
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
    dump_data_and_model(X, model, model_onnx, folder=folder, allow_failure=allow_failure,
                        basename=prefix + "Mcl" + model.__class__.__name__ + suffix)


def dump_multiple_regression(model, suffix="", folder=None, allow_failure=None):
    """
    Trains and dumps a model for a multi regression problem.

    :param model: any model following *scikit-learn* API
    :param suffix: added to filenames
    :param folder: where to save the file
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :return: output of :func:`dump_data_and_model`

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([[100, 50], [100, 49], [100, 99]], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, 'tree-based multi-output regressor', [('input', FloatTensorType([1, 2]))])
    dump_data_and_model(X, model, model_onnx, folder=folder, allow_failure=allow_failure,
                        basename=prefix + "MRg" + model.__class__.__name__ + suffix)


def dump_single_regression(model, suffix="", folder=None, allow_failure=None):
    """
    Trains and dumps a model for a regression problem.

    :param model: any model following *scikit-learn* API
    :param prefix: library name
    :param suffix: added to filenames
    :param folder: where to save the file
    :param allow_failure: None to raise an exception if comparison fails
        for the backends, otherwise a string which is then evaluated to check
        whether or not the test can fail, example:
        ``"StrictVersion(onnx.__version__) < StrictVersion('1.3.0')"``
    :return: output of :func:`dump_data_and_model`

    Every created filename will follow the pattern:
    ``<folder>/<prefix><task><classifier-name><suffix>.<data|expected|model|onnx>.<pkl|onnx>``.
    """
    X = [[0, 1], [1, 1], [2, 0]]
    X = numpy.array(X, dtype=numpy.float32)
    y = numpy.array([100, -10, 50], dtype=numpy.float32)
    model.fit(X, y)
    model_onnx, prefix = convert_model(model, 'tree-based regressor', [('input', FloatTensorType([1, 2]))])
    dump_data_and_model(X, model, model_onnx, folder=folder, allow_failure=allow_failure,
                        basename=prefix + "Reg" + model.__class__.__name__ + suffix)


def make_report_backend(folder):
    """
    Looks into a folder for dumped files after
    the unit tests.
    """
    res = {}
    files = os.listdir(folder)
    for name in files:
        if name.endswith(".expected.pkl"):
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model]["_tested"] = True
        elif '.backend.' in name:
            bk = name.split(".backend.")[-1].split(".")[0]
            model = name.split(".")[0]
            if model not in res:
                res[model] = {}
            res[model][bk] = True

    def dict_update(d, u):
        d.update(u)
        return d

    aslist = [dict_update(dict(_model=k), v) for k, v in res.items()]
    return aslist
