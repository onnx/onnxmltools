# SPDX-License-Identifier: Apache-2.0

from onnx.defs import onnx_opset_version

DEFAULT_OPSET_NUMBER = 15


def _get_maximum_opset_supported():
    return min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def get_maximum_opset_supported():
    return min(
        onnx_opset_version(), max(DEFAULT_OPSET_NUMBER, _get_maximum_opset_supported())
    )
