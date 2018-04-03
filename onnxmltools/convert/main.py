# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .common import utils


def convert_sklearn(model, name=None, input_features=None):
    if not utils.sklearn_installed():
        raise RuntimeError('scikit-learn is not installed. Please install scikit-learn to use this feature.')

    from .sklearn.convert import convert
    return convert(model, name, input_features)


def convert_coreml(model, name=None):
    if not utils.coreml_installed():
        raise RuntimeError('coremltools is not installed. Please install coremltools to use this feature.')

    from .coreml.convert import convert
    return convert(model, name)


def convert_keras(model, name=None):
    if not utils.keras_installed():
        raise RuntimeError('keras is not installed. Please install coremltools to use this feature.')

    from .keras.convert import convert
    return convert(model, name)
