# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter


def convert_one_hot_encoder(scope, operator, container):
    op_type = "OneHotEncoder"
    attrs = {"name": operator.full_name}

    raw_model = operator.raw_operator.oneHotEncoder
    if raw_model.HasField("int64Categories"):
        attrs["cats_int64s"] = list(int(i) for i in raw_model.int64Categories.vector)
    if raw_model.HasField("stringCategories"):
        attrs["cats_strings"] = list(
            str(s).encode("utf-8") for s in raw_model.stringCategories.vector
        )

    container.add_node(
        op_type,
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name],
        op_domain="ai.onnx.ml",
        **attrs
    )


register_converter("oneHotEncoder", convert_one_hot_encoder)
