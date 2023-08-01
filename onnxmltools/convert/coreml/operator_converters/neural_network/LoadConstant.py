# SPDX-License-Identifier: Apache-2.0

from .....proto import helper
from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_constant

def convert_load_constant(scope, operator, container):
    params = operator.raw_operator.loadConstant
    constant_name = scope.get_unique_variable_name('constant')
    constant = helper.make_tensor(constant_name, onnx_proto.TensorProto.FLOAT,
                                  params.shape, params.data.floatValue)

    apply_constant(scope, operator.output_full_names, container,
                   operator_name=operator.full_name, value=constant)

register_converter('loadConstant', convert_load_constant)
