# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
The entry point to onnxconverter-common.
This framework performs optimization for ONNX models and
includes common utilities for ONNX converters.
"""
__version__ = "1.4.2"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxconverter-common"
__model_version__ = 0

from .optimizer import optimize_onnx
from .onnx_ops import *
from .container import *
from .registration import *
from .topology import *
from .case_insensitive_dict import *
from .data_types import *
from .interface import *
from .shape_calculator import *
from .tree_ensemble import *
from .utils import *
