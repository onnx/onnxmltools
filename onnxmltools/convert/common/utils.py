# SPDX-License-Identifier: Apache-2.0
import warnings
import numpy as np


def hummingbird_installed():
    """
    Checks that *Hummingbird* is available.
    """
    try:
        import hummingbird.ml  # noqa: F401

        return True
    except ImportError:
        return False


def tf2onnx_installed():
    """
    Checks that *tf2onnx* is available.
    """
    try:
        import tf2onnx  # noqa F401

        return True
    except ImportError:
        return False


def sparkml_installed():
    """
    Checks that *spark* is available.
    """
    try:
        import pyspark  # noqa F401

        return True
    except ImportError:
        return False


def sklearn_installed():
    """
    Checks that *scikit-learn* is available.
    """
    try:
        import sklearn  # noqa F401

        return True
    except ImportError:
        return False


def skl2onnx_installed():
    """
    Checks that *skl2onnx* converter is available.
    """
    try:
        import skl2onnx  # noqa F401

        return True
    except ImportError:
        return False


def coreml_installed():
    """
    Checks that *coremltools* is available.
    """
    try:
        import coremltools  # noqa F401

        return True
    except ImportError:
        return False


def keras2onnx_installed():
    """
    Checks that *keras2onnx* is available.
    """
    try:
        import keras2onnx  # noqa F401

        return True
    except ImportError:
        return False


def torch_installed():
    """
    Checks that *pytorch* is available.
    """
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


def caffe2_installed():
    """
    Checks that *caffe* is available.
    """
    try:
        import caffe2  # noqa F401

        return True
    except ImportError:
        return False


def libsvm_installed():
    """
    Checks that *libsvm* is available.
    """
    try:
        import svm  # noqa F401
        import svmutil  # noqa F401

        return True
    except ImportError:
        return False


def lightgbm_installed():
    """
    Checks that *lightgbm* is available.
    """
    try:
        import lightgbm  # noqa F401

        return True
    except ImportError:
        return False


def xgboost_installed():
    """
    Checks that *xgboost* is available.
    """
    try:
        import xgboost  # noqa F401
    except ImportError:
        return False
    from xgboost.core import _LIB

    try:
        _LIB.XGBoosterDumpModelEx
    except AttributeError:
        # The version is not recent enough even though it is version 0.6.
        # You need to install xgboost from github and not from pypi.
        return False
    import packaging.version as pv
    from xgboost import __version__

    vers = pv.Version(__version__)
    allowed = pv.Version("0.7")
    if vers < allowed:
        warnings.warn(
            "The converter works for xgboost >= 0.7. Earlier versions might not."
        )
    return True


def h2o_installed():
    """
    Checks that *h2o* is available.
    """
    try:
        import h2o  # noqa F401
    except ImportError:
        return False
    return True


def get_producer():
    """
    Internal helper function to return the producer
    """
    from ... import __producer__

    return __producer__


def get_producer_version():
    """
    Internal helper function to return the producer version
    """
    from ... import __producer_version__

    return __producer_version__


def get_domain():
    """
    Internal helper function to return the model domain
    """
    from ... import __domain__

    return __domain__


def get_model_version():
    """
    Internal helper function to return the model version
    """
    from ... import __model_version__

    return __model_version__


def is_numeric_type(item):
    numeric_types = (int, float, complex)
    types = numeric_types

    if isinstance(item, list):
        return all(isinstance(i, types) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.number)
    return isinstance(item, types)


def is_string_type(item):
    if isinstance(item, list):
        return all(isinstance(i, str) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.str_)
    return isinstance(item, str)
