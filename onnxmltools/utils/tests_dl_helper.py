#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------
"""
Helpers to help testing deep learning converted models.
"""
import numpy as np
from .main import save_model

rt_onnxruntime, rt_caffe2, rt_cntk = range(3)
_rt_installed = None


def find_inference_engine():
    """
    Finds an available backend to evaluate deep learning models.
    Returns its name as a string.
    """
    global _rt_installed
    if _rt_installed is not None:
        return _rt_installed

    try:
        import onnxruntime
        _rt_installed = rt_onnxruntime
    except ImportError:
        try:
            import cntk
            _rt_installed = rt_cntk
        except ImportError:
            try:
                import caffe2
                _rt_installed = rt_caffe2
            except ImportError:
                pass

    return _rt_installed


def evaluate_deep_model(onnx_model, inputs, rt_type=None):
    """
    Evaluates a deep learning model with
    onnxruntime, *CNTK* or *caffe2*
    """
    if rt_type is None:
        rt_type = find_inference_engine()
    if rt_type == rt_onnxruntime:
        return _evaluate_onnxruntime(onnx_model, inputs)
    if rt_type == rt_cntk:
        return _evaluate_cntk(onnx_model, inputs)
    elif rt_type == rt_caffe2:
        return _evaluate_caffe2(onnx_model, inputs)
    else:
        raise ImportError('No runtime found. Need either CNTK or Caffe2')


def _evaluate_onnxruntime(onnx_model, inputs):
    import onnxruntime
    runtime = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    result = None
    inputs = inputs if isinstance(inputs, list) else [inputs]
    for i_ in range(inputs[0].shape[0]):  # TODO: onnxruntime can't support batch_size > 1
        out = runtime.run([], {x.name: inputs[n_][i_:i_ + 1] for n_, x in enumerate(runtime.get_inputs())})[0]
        result = out if result is None else np.concatenate((result, out))

    return result[0] if isinstance(result, list) else result


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
    save_model(onnx_model, temporary_onnx_model_file_name)
    cntk_model = cntk.Function.load(temporary_onnx_model_file_name, format=cntk.ModelFormat.ONNX)

    return cntk_model.eval(adjusted_inputs)


def create_tensor(N, C, H=None, W=None):
    if H is None and W is None:
        return np.random.rand(N, C).astype(np.float32, copy=False)
    elif H is not None and W is not None:
        return np.random.rand(N, C, H, W).astype(np.float32, copy=False)
    else:
        raise ValueError('This function only produce 2-D or 4-D tensor')
