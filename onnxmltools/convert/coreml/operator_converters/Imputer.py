# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter


def convert_imputer(scope, operator, container):
    op_type = "Imputer"
    attrs = {"name": operator.full_name}
    imputer = operator.raw_operator.imputer
    if imputer.HasField("replaceDoubleValue"):
        attrs["replaced_value_float"] = imputer.replaceDoubleValue
    elif imputer.HasField("replaceInt64Value"):
        attrs["replaced_value_int64"] = imputer.replaceInt64Value
    if imputer.HasField("imputedDoubleArray"):
        attrs["imputed_value_floats"] = imputer.imputedDoubleArray.vector
    elif imputer.HasField("imputedInt64Array"):
        attrs["imputed_value_int64s"] = imputer.imputedInt64Array.vector
    container.add_node(
        op_type,
        operator.input_full_names,
        operator.output_full_names,
        op_domain="ai.onnx.ml",
        **attrs
    )


register_converter("imputer", convert_imputer)
