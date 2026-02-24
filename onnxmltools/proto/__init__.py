# SPDX-License-Identifier: Apache-2.0

# Rather than using ONNX protobuf definition throughout our codebase,
# we import ONNX protobuf definition here so that
# we can conduct quick fixes by overwriting ONNX functions without
# changing any lines elsewhere.
from onnx import onnx_pb as onnx_proto
from onnx import helper

from onnx.onnx_pb import TensorProto
