#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

"""
Helpers for converters.
"""
import numbers
import six
import numpy as np
import warnings
from distutils.version import LooseVersion


def sklearn_installed():
    """
    Checks that *scikit-learn* is available.
    """
    try:
        import sklearn
        return True
    except ImportError:
        return False


def coreml_installed():
    """
    Checks that *coremltools* is available.
    """
    try:
        import coremltools
        return True
    except ImportError:
        return False


def keras_installed():
    """
    Checks that *keras* is available.
    """
    try:
        import keras
        return True
    except ImportError:
        return False


def torch_installed():
    """
    Checks that *pytorch* is available.
    """
    try:
        import torch
        return True
    except ImportError:
        return False
    try:
        import torch.onnx
    except ImportError:
        return False


def caffe2_installed():
    """
    Checks that *caffe* is available.
    """
    try:
        import caffe2
        return True
    except ImportError:
        return False
    try:
        import onnx_caffe2
    except ImportError:
        return False


def libsvm_installed():
    """
    Checks that *libsvm* is available.
    """
    try:
        import svm
        import svmutil
        return True
    except ImportError:
        return False


def xgboost_installed():
    """
    Checks that *xgboost* is available.
    """
    try:
        import xgboost
    except ImportError:
        return False
    from xgboost.core import _LIB
    try:
        _LIB.XGBoosterDumpModelEx
    except AttributeError:
        # The version is now recent enough even though it is version 0.6.
        # You need to install xgboost from github and not from pypi.
        return False
    from xgboost import __version__
    vers = LooseVersion(__version__)
    allowed = LooseVersion('0.7')
    if vers < allowed:
        warnings.warn('The converter works for xgboost >= 0.7. Earlier versions might not.')
    return True


def is_numeric_type(item):
    if six.PY2:
        numeric_types = (int, float, long, complex)
    else:
        numeric_types = (int, float, complex)
    types = numeric_types

    if isinstance(item, list):
        return all(isinstance(i, types) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.number)
    return isinstance(item, types)


def is_string_type(item):
    types = (six.string_types, six.text_type)
    if isinstance(item, list):
        return all(isinstance(i, types) for i in item)
    if isinstance(item, np.ndarray):
        return np.issubdtype(item.dtype, np.str_)
    return isinstance(item, types)


def _check_has_attr(obj, attribute):
    if not hasattr(obj, attribute):
        raise AttributeError(attribute)


def cast_list(type, items):
    return [type(item) for item in items]


def convert_to_python_value(var):
    import numbers
    if isinstance(var, numbers.Integral):
        return int(var)
    elif isinstance(var, numbers.Real):
        return float(var)
    elif isinstance(var, str):
        return str(var)
    else:
        raise TypeError('Unable to convert {0} to python type'.format(type(var)))


def convert_to_python_default_value(var):
    if isinstance(var, numbers.Integral):
        return int()
    elif isinstance(var, numbers.Real):
        return float()
    elif isinstance(var, str):
        return str()
    else:
        raise TypeError('Unable to find default python value for type {0}'.format(type(var)))


def convert_to_list(var):
    if isinstance(var, numbers.Real) or isinstance(var, str):
        return [convert_to_python_value(var)]
    elif isinstance(var, np.ndarray) and len(var.shape) == 1:
        return [convert_to_python_value(v) for v in var]
    elif isinstance(var, list):
        flattened = []
        if all(isinstance(ele, np.ndarray) and len(ele.shape) == 1 for ele in var):
            max_classes = max([ele.shape[0] for ele in var])
            flattened_one = []
            for ele in var:
                for i in range(max_classes):
                    if i < ele.shape[0]:
                        flattened_one.append(convert_to_python_value(ele[i]))
                    else:
                        flattened_one.append(convert_to_python_default_value(ele[0]))
            flattened += flattened_one
            return flattened
        elif all(isinstance(v, numbers.Real) or isinstance(v, str) for v in var):
            return [convert_to_python_value(v) for v in var]
        else:
            raise TypeError('Unable to flatten variable')
    else:
        raise TypeError('Unable to flatten variable')
