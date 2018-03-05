# Copyright (c) 2017, Microsoft. All rights reserved.
#
# See LICENSE.txt.
"""
Main entry point to onnxmltools.
This framework converts any machine learned model into onnx format
which is a common language to describe any machine learned model.
"""
__version__ = "0.1.0.0000"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxml"
__model_version__ = 0


from .convert import convert_coreml
from .convert import convert_sklearn

from .utils import load_model
from .utils import save_model
from .utils import save_text

