# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter


def convert_scaler(scope, operator, container):
    op_type = "Scaler"
    attrs = {"name": operator.full_name}
    scaler = operator.raw_operator.scaler

    scale = [x for x in scaler.scaleValue]
    offset = [-x for x in scaler.shiftValue]

    attrs["scale"] = scale
    attrs["offset"] = offset

    container.add_node(
        op_type,
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        **attrs
    )


register_converter("scaler", convert_scaler)
