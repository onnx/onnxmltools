# SPDX-License-Identifier: Apache-2.0

from onnx.defs import onnx_opset_version

DEFAULT_OPSET_NUMBER = 15
OPSET_TO_IR_VERSION = {
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 7,
    14: 7,
    15: 8,
    16: 8,
    17: 8,
    18: 8,
    19: 9,
    20: 9,
    21: 10,
    22: 10,
    23: 10,
    24: 10,
}


def _get_maximum_opset_supported():
    return min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


def get_maximum_opset_supported():
    return min(
        onnx_opset_version(), max(DEFAULT_OPSET_NUMBER, _get_maximum_opset_supported())
    )
