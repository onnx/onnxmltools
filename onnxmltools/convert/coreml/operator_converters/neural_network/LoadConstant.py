# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import helper
from .....proto import onnx_proto
from ....common._registration import register_converter
from ...common._apply_operation import apply_constant

def convert_load_constant(scope, operator, container):
    params = operator.raw_operator.loadConstant
    constant_name = scope.get_unique_variable_name('constant')
    constant = helper.make_tensor(constant_name, onnx_proto.TensorProto.FLOAT,
                                  params.shape, params.data.floatValue)

    apply_constant(scope, [], operator.output_full_names, op_version=container.target_opset,
                   operator_name=operator.full_name, value=constant)

register_converter('loadConstant', convert_load_constant)
