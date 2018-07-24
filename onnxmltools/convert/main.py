# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..proto import onnx
from .common import utils


def convert_sklearn(model, name=None, initial_types=None, doc_string='',
                    targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.sklearn_installed():
        raise RuntimeError('scikit-learn is not installed. Please install scikit-learn to use this feature.')

    from .sklearn.convert import convert
    return convert(model, name, initial_types,
                   doc_string, targeted_onnx, custom_conversion_functions, custom_shape_calculators)


def convert_coreml(model, name=None, initial_types=None, doc_string='',
                   targeted_onnx=onnx.__version__ , custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.coreml_installed():
        raise RuntimeError('coremltools is not installed. Please install coremltools to use this feature.')

    from .coreml.convert import convert
    return convert(model, name, initial_types,
                   doc_string, targeted_onnx, custom_conversion_functions, custom_shape_calculators)


def convert_keras(model, name=None, initial_types=None, doc_string='',
                  targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    if not utils.keras_installed():
        raise RuntimeError('keras is not installed. Please install it to use this feature.')

    from .keras.convert import convert
    return convert(model, name, initial_types,
                   doc_string, targeted_onnx, custom_conversion_functions, custom_shape_calculators)
