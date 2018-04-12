# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# Rather than using ONNX protobuf definition throughout our codebase, we import ONNX protobuf definition here so that
# we can conduct quick fixes by overwriting ONNX functions here without changing any lines elsewhere.
from onnx import onnx_ml_pb2 as onnx_proto
from onnx import helper
