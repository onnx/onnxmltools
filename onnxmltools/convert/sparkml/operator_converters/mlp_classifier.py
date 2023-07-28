# SPDX-License-Identifier: Apache-2.0

from pyspark.ml.classification import MultilayerPerceptronClassificationModel

from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._topology import Operator, Scope, ModelComponentContainer
from ....proto import onnx_proto
from typing import List
import numpy as np


def convert_sparkml_mlp_classifier(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op: MultilayerPerceptronClassificationModel = operator.raw_operator
    layers: List[int] = op.getLayers()
    weights: np.ndarray = op.weights.toArray()

    offset = 0

    input: str
    for i in range(len(layers) - 1):
        weight_matrix = weights[offset : offset + layers[i] * layers[i + 1]].reshape(
            [layers[i], layers[i + 1]]
        )
        offset += layers[i] * layers[i + 1]
        bias_vector = weights[offset : offset + layers[i + 1]]
        offset += layers[i + 1]

        if i == 0:
            input = operator.inputs[0].full_name

        weight_variable = scope.get_unique_variable_name("w")
        container.add_initializer(
            weight_variable,
            onnx_proto.TensorProto.FLOAT,
            weight_matrix.shape,
            weight_matrix.flatten().astype(np.float32),
        )

        bias_variable = scope.get_unique_variable_name("b")
        container.add_initializer(
            bias_variable,
            onnx_proto.TensorProto.FLOAT,
            bias_vector.shape,
            bias_vector.astype(np.float32),
        )

        gemm_output_variable = scope.get_unique_variable_name("gemm_output")
        container.add_node(
            op_type="Gemm",
            inputs=[input, weight_variable, bias_variable],
            outputs=[gemm_output_variable],
            op_version=7,
            name=scope.get_unique_operator_name("Gemm"),
        )

        if i == len(layers) - 2:
            container.add_node(
                op_type="Softmax",
                inputs=[gemm_output_variable],
                outputs=[operator.outputs[1].full_name],
                op_version=1,
                name=scope.get_unique_operator_name("Softmax"),
            )
        else:
            input = scope.get_unique_variable_name("activation_output")
            container.add_node(
                op_type="Sigmoid",
                inputs=[gemm_output_variable],
                outputs=[input],
                op_version=1,
                name=scope.get_unique_operator_name("Sigmoid"),
            )

    container.add_node(
        "ArgMax",
        [operator.outputs[1].full_name],
        [operator.outputs[0].full_name],
        name=scope.get_unique_operator_name("ArgMax"),
        axis=1,
        keepdims=0,
    )


register_converter(
    "pyspark.ml.classification.MultilayerPerceptronClassificationModel",
    convert_sparkml_mlp_classifier,
)


def calculate_mlp_classifier_output_shapes(operator: Operator):
    op: MultilayerPerceptronClassificationModel = operator.raw_operator

    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType]
    )

    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Input must be a [N, C]-tensor")

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])
    class_count = op.numClasses
    operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator(
    "pyspark.ml.classification.MultilayerPerceptronClassificationModel",
    calculate_mlp_classifier_output_shapes,
)
