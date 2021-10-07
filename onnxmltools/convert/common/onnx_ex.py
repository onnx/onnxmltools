# SPDX-License-Identifier: Apache-2.0

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import (
    get_maximum_opset_supported as _get_maximum_opset_supported)


def get_maximum_opset_supported():
    return min(onnx_opset_version(),
               max(15, _get_maximum_opset_supported()))
