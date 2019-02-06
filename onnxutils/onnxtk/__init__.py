# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
The entry point to onnxtk.
This framework performs optimization for onnx model.
"""
__version__ = "0.0.1"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxtk"
__model_version__ = 0

from .optimizer import optimize_onnx
