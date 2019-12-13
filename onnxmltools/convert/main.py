# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx
from .common import utils
import warnings

def convert_coreml(model, name=None, initial_types=None, doc_string='', target_opset=None,
                   targeted_onnx=onnx.__version__ , custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.coreml_installed():
        raise RuntimeError('coremltools is not installed. Please install coremltools to use this feature.')

    from .coreml.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_keras(model, name=None, initial_types=None, doc_string='',
                  target_opset=None, targeted_onnx=onnx.__version__,
                  channel_first_inputs=None, custom_conversion_functions=None, custom_shape_calculators=None,
                  default_batch_size=1):
    if not utils.keras2onnx_installed():
        raise RuntimeError('keras2onnx is not installed. Please install it to use this feature.')

    if custom_conversion_functions:
        warnings.warn('custom_conversion_functions is not supported any more. Please set it to None.')

    from keras2onnx import convert_keras as convert
    return convert(model, name, doc_string, target_opset, channel_first_inputs)


def convert_libsvm(model, name=None, initial_types=None, doc_string='', target_opset=None,
                     targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.libsvm_installed():
        raise RuntimeError('libsvm is not installed. Please install libsvm to use this feature.')

    from .libsvm.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_lightgbm(model, name=None, initial_types=None, doc_string='', target_opset=None,
                     targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.lightgbm_installed():
        raise RuntimeError('lightgbm is not installed. Please install lightgbm to use this feature.')

    from .lightgbm.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators)


def convert_sklearn(model, name=None, initial_types=None, doc_string='', target_opset=None,
                    targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.sklearn_installed():
        raise RuntimeError('scikit-learn is not installed. Please install scikit-learn to use this feature.')

    if not utils.skl2onnx_installed():
        raise RuntimeError('skl2onnx is not installed. Please install skl2onnx to use this feature.')

    from skl2onnx.convert import convert_sklearn as convert_skl2onnx
    return convert_skl2onnx(model, name, initial_types, doc_string, target_opset,
                   custom_conversion_functions, custom_shape_calculators)

def convert_sparkml(model, name=None, initial_types=None, doc_string='', target_opset=None,
                    targeted_onnx=onnx.__version__, custom_conversion_functions=None,
                    custom_shape_calculators=None, spark_session=None):
    if not utils.sparkml_installed():
        raise RuntimeError('Spark is not installed. Please install Spark to use this feature.')

    from .sparkml.convert import convert
    return convert(model, name, initial_types, doc_string, target_opset, targeted_onnx,
                   custom_conversion_functions, custom_shape_calculators, spark_session)

def convert_tensorflow(frozen_graph_def,
                       name=None, input_names=None, output_names=None,
                       doc_string='',
                       target_opset=None,
                       channel_first_inputs=None,
                       debug_mode=False, custom_op_conversions=None):
    if not utils.keras2onnx_installed():
        raise RuntimeError('keras2onnx is not installed. Please install it to use this feature.')

    from keras2onnx import convert_tensorflow as convert
    return convert(frozen_graph_def, name, input_names, output_names, doc_string,
                   target_opset, channel_first_inputs, debug_mode, custom_op_conversions)

def convert_xgboost(*args, **kwargs):
    if not utils.xgboost_installed():
        raise RuntimeError('xgboost is not installed. Please install xgboost to use this feature.')

    from .xgboost.convert import convert
    return convert(*args, **kwargs)

def convert_h2o(*args, **kwargs):
    if not utils.h2o_installed():
        raise RuntimeError('h2o is not installed. Please install h2o to use this feature.')

    from .h2o.convert import convert
    return convert(*args, **kwargs)
