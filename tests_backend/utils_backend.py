"""
Helpers to test runtimes.
"""
import os
import glob
import pickle
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal

class ExpectedAssertionError(Exception):
    """
    Expected failure.
    """
    pass


class OnnxRuntimeAssertionError(AssertionError):
    """
    Expected failure.
    """
    pass


def search_converted_models(root=None):
    """
    Searches for all converted models generated by
    unit tests in folders tests and with function
    *dump_data_and_model*.
    """
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
        root = os.path.normpath(root)
    if not os.path.exists(root):
        raise FileNotFoundError("Unable to find '{0}'.".format(root))
    
    founds = glob.iglob(f"{root}/**/*.model.onnx", recursive=True)
    keep = []
    for found in founds:
        onnx = found
        basename = onnx[:-len(".model.onnx")]
        data = basename + ".data.pkl"
        model = basename + ".model.pkl"
        expected = basename + ".expected.pkl"
        res = dict(onnx=onnx, data=data, model=model, expected=expected)
        ok = True
        for k, v in res.items():
            if not os.path.exists(v):
                ok = False
        if ok:
            keep.append((basename, res))
    keep.sort()
    return [_[1] for _ in keep]
    

def load_data_and_model(items_as_dict):
    """
    Loads every file in a dictionary with extension pkl
    for pickle.
    """
    res = {}
    for k, v in items_as_dict.items():
        if os.path.splitext(v)[-1] == ".pkl":
            with open(v, "rb") as f:
                res[k] = pickle.load(f)
        else:
            res[k] = v
    return res


def extract_options(name):
    """
    Extracts comparison option from filename.
    As example, ``Binarizer-SkipDim1`` means
    options *SkipDim1* is enabled. 
    ``(1, 2)`` and ``(2,)`` are considered equal.
    Available options:
    
    * `'SkipDim1'`: reshape arrays by skipping 1-dimension: ``(1, 2)`` --> ``(2,)``
    * `'OneOff'`: inputs comes in a list for the predictions are computed with a call for each of them,
        not with one call
    """
    opts = name.replace("\\", "/").split("/")[-1].split('.')[0].split('-')
    if len(opts) == 1:
        return {}
    else:
        res = {}
        for opt in opts[1:]:
            if opt in ("SkipDim1", "OneOff", "NoProb", "Dec4", "Dec3",
                       "Disc", "Mism", "CannotLoad", "Fail"):
                res[opt] = True
            else:
                raise NameError("Unable to parse option '{}'".format(opts[1:]))
        return res


def compare(expected, output, **kwargs):
    """
    Compares expected values and output.
    Returns None if no error, an exception message otherwise.
    """
    SkipDim1 = kwargs.pop("SkipDim1", False)
    NoProb = kwargs.pop("NoProb", False)
    Dec4 = kwargs.pop("Dec4", False)
    Dec3 = kwargs.pop("Dec3", False)
    Disc = kwargs.pop("Disc", False)
    Mism = kwargs.pop("Mism", False)
    
    if Dec4:
        kwargs["decimal"] = min(kwargs["decimal"], 4)
    if Dec3:
        kwargs["decimal"] = min(kwargs["decimal"], 3)
    if isinstance(expected, numpy.ndarray) and isinstance(output, numpy.ndarray):
        if SkipDim1:
            # Arrays like (2, 1, 2, 3) becomes (2, 2, 3) as one dimension is useless.
            expected = expected.reshape(tuple([d for d in expected.shape if d > 1]))
            output = output.reshape(tuple([d for d in expected.shape if d > 1]))
        if NoProb:
            # One vector is (N,) with scores, negative for class 0
            # positive for class 1
            # The other vector is (N, 2) score in two columns.
            if len(output.shape) == 2 and output.shape[1] == 2 and len(expected.shape) == 1:
                output = output[:, 1]
            elif len(output.shape) == 1 and len(expected.shape) == 1:
                pass
            elif len(expected.shape) == 1 and len(output.shape) == 2 and \
                    expected.shape[0] == output.shape[0] and output.shape[1] == 1:
                output = output[:, 0]
            elif expected.shape != output.shape:
                raise NotImplementedError("No good shape: {0} != {1}".format(expected.shape, output.shape))
        if len(expected.shape) == 1 and len(output.shape) == 2 and output.shape[1] == 1:
            output = output.ravel()
        if expected.dtype in (numpy.str, numpy.dtype("<U1")):
            try:
                assert_array_equal(expected, output)
            except Exception as e:
                if Disc:
                    # Bug to be fixed later.
                    return ExpectedAssertionError(str(e))
                else:
                    return OnnxRuntimeAssertionError(str(e))
        else:
            try:
                assert_array_almost_equal(expected, output, **kwargs)
            except Exception as e:
                expected_ = expected.ravel()
                output_ = output.ravel()
                if len(expected_) == len(output_):
                    diff = numpy.abs(expected_ - output_).max()
                elif Mism:
                    return ExpectedAssertionError("dimension mismatch={0}, {1}\n{2}".format(expected.shape, output.shape, e))
                else:
                    return OnnxRuntimeAssertionError("dimension mismatch={0}, {1}\n{2}".format(expected.shape, output.shape, e))
                if Disc:
                    # Bug to be fixed later.
                    return ExpectedAssertionError("max diff(expected, output)={0}\n{1}".format(diff, e))
                else:
                    return OnnxRuntimeAssertionError("max diff(expected, output)={0}\n{1}".format(diff, e))
    else:
        return OnnxRuntimeAssertionError("Unexpected types {0} != {1}".format(type(expected), type(output)))
    return None
