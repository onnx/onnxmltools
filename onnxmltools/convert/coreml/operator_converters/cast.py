# SPDX-License-Identifier: Apache-2.0

from ...common._apply_operation import apply_cast
from ...common._registration import register_converter


def convert_cast(scope, operator, container):
    target_type = operator.outputs[0].type.to_onnx_type().tensor_type.elem_type
    apply_cast(
        scope,
        operator.input_full_names,
        operator.output_full_names,
        container,
        operator_name=operator.full_name,
        to=target_type,
    )


register_converter("cast", convert_cast)
