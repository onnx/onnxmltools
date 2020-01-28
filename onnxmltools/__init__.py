# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Main entry point to onnxmltools.
This framework converts any machine learned model into onnx format
which is a common language to describe any machine learned model.
"""
__version__ = "1.6.5"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxml"
__model_version__ = 0


from .convert import convert_coreml
from .convert import convert_keras
from .convert import convert_lightgbm
from .convert import convert_sklearn
from .convert import convert_sparkml
from .convert import convert_tensorflow
from .convert import convert_xgboost
from .convert import convert_h2o

from .utils import load_model
from .utils import save_model
