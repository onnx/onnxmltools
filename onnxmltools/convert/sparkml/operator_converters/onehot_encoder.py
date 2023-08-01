# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter, register_shape_calculator
from ....proto import onnx_proto
from ...common.data_types import FloatTensorType
from ...common._topology import Operator, Scope
from pyspark.ml.feature import OneHotEncoderModel
from typing import List


def _get_categories(operator: Operator) -> List[List[int]]:
    op: OneHotEncoderModel = operator.raw_operator
    categorySizes: List[int] = op.categorySizes

    # This is necessary to match SparkML's OneHotEncoder behavior.
    # If handleInvalid is set to "keep", an extra "category" indicating
    # invalid values is added as last category:
    #   - if dropLast is set to false, then an extra bit will be added for invalid input,
    #     which does not match ONNX's OneHotEncoder behavior.
    #   - if dropLast is set to true, then invalid values are encoded as all-zeros vector,
    #     which matches ONNX's OneHotEncoder behavior when "zeros" is set to 1.
    # If handleInvalid is set to "error", then:
    #   - if dropLast is set to false, then nothing will be added or dropped,
    #     and matches ONNX's OneHotEncoder behavior when "zeros" is set to 0.
    #   - if dropLast is set to true, then the last bit will be dropped,
    #     which does not match ONNX's OneHotEncoder behavior.
    if (op.getHandleInvalid() == "keep" and not op.getDropLast()) or (
        op.getHandleInvalid() == "error" and op.getDropLast()
    ):
        raise RuntimeError(
            f"The 'handleInvalid' and 'dropLast' parameters must be set to "
            f"('keep', True) or ('error', False), but got "
            f"('{op.getHandleInvalid()}', {op.getDropLast()}) instead."
        )

    return [list(range(0, size)) for size in categorySizes]


def convert_sparkml_one_hot_encoder(scope: Scope, operator: Operator, container):
    categories = _get_categories(operator)
    N = operator.inputs[0].type.shape[0] or -1

    zeros = 1 if operator.raw_operator.getHandleInvalid() == "keep" else 0

    for i in range(0, len(categories)):
        encoder_type = "OneHotEncoder"

        # Set zeros to 0 to match the "error"
        # handleInvalid behavior of SparkML's OneHotEncoder.
        encoder_attrs = {
            "name": scope.get_unique_operator_name(encoder_type),
            "cats_int64s": categories[i],
            "zeros": zeros,
        }

        encoded_feature_name = scope.get_unique_variable_name(
            "encoded_feature_at_" + str(i)
        )
        container.add_node(
            op_type=encoder_type,
            inputs=[operator.inputs[i].full_name],
            outputs=[encoded_feature_name],
            op_domain="ai.onnx.ml",
            op_version=1,
            **encoder_attrs,
        )

        shape_variable_name = scope.get_unique_variable_name("shape_at_" + str(i))
        container.add_initializer(
            shape_variable_name,
            onnx_proto.TensorProto.INT64,
            [2],
            [N, len(categories[i])],
        )

        container.add_node(
            op_type="Reshape",
            inputs=[encoded_feature_name, shape_variable_name],
            outputs=[operator.outputs[i].full_name],
            op_domain="ai.onnx",
            op_version=13,
        )


register_converter(
    "pyspark.ml.feature.OneHotEncoderModel", convert_sparkml_one_hot_encoder
)


def calculate_sparkml_one_hot_encoder_output_shapes(operator: Operator):
    categories = _get_categories(operator)
    N = operator.inputs[0].type.shape[0]
    for i, output in enumerate(operator.outputs):
        output.type = FloatTensorType([N, len(categories[i])])


register_shape_calculator(
    "pyspark.ml.feature.OneHotEncoderModel",
    calculate_sparkml_one_hot_encoder_output_shapes,
)
