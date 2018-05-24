import unittest
import coremltools
import onnxmltools
import numpy as np
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.decomposition import TruncatedSVD

np.random.seed(0)


def _find_backend():
    try:
        import cntk
        return 'cntk'
    except:
        pass
    try:
        import caffe2
        return 'caffe2'
    except:
        pass
    return None


def _evaluate(onnx_model, inputs):
    runtime_name = _find_backend()
    if runtime_name == 'cntk':
        return _evaluate_cntk(onnx_model, inputs)
    elif runtime_name == 'caffe2':
        return _evaluate_caffe2(onnx_model, inputs)
    else:
        raise RuntimeError('No runtime found. Need either CNTK or Caffe2')


def _evaluate_caffe2(onnx_model, inputs):
    from caffe2.python.onnx.backend import run_model

    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = run_model(onnx_model, inputs)

    adjusted_outputs = dict()
    for output in onnx_model.graph.output:
        adjusted_outputs[output.name] = outputs[output.name]

    return adjusted_outputs[onnx_model.graph.output[0].name]


def _evaluate_cntk(onnx_model, inputs):
    import cntk
    if not isinstance(inputs, list):
        inputs = [inputs]

    adjusted_inputs = dict()
    for i, x in enumerate(inputs):
        onnx_name = onnx_model.graph.input[i].name
        adjusted_inputs[onnx_name] = [np.ascontiguousarray(np.squeeze(_, axis=0)) for _ in np.split(x, x.shape[0])]

    temporary_onnx_model_file_name = 'temp_' + onnx_model.graph.name + '.onnx'
    onnxmltools.utils.save_model(onnx_model, temporary_onnx_model_file_name)
    cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

    return cntk_model.eval(adjusted_inputs)


def _create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor')


class TestSklearn2ONNX(unittest.TestCase):

    def test_truncated_svd(self):
        N, C, K = 2, 3, 2
        x = _create_tensor(N, C)

        svd = TruncatedSVD(n_components=K)
        svd.fit(x)
        onnx_model = onnxmltools.convert_sklearn(svd, initial_types=[('input', FloatTensorType(shape=[1, C]))])
        y_reference = svd.transform(x)
        y_produced = _evaluate(onnx_model, x)

        self.assertTrue(np.allclose(y_reference, y_produced))
