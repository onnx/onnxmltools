# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import StringTensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._topology import Operator, Scope, ModelComponentContainer
from pyspark.ml.feature import CountVectorizerModel


def convert_count_vectorizer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    op: CountVectorizerModel = operator.raw_operator
    vocab, minTF, binary = (
        op.vocabulary,
        op.getOrDefault("minTF"),
        op.getOrDefault("binary"),
    )

    if minTF < 1.0:
        raise NotImplementedError(
            "Converting to ONNX for CountVectorizerModel is not supported when minTF < 1.0"
        )

    min_opset = 9
    if not binary:
        # If binary is False, then we need the ThresholdedRelu
        # operator which is only available since opset 10.
        min_opset = 10

    if container.target_opset < min_opset:
        raise NotImplementedError(
            f"Converting to ONNX for CountVectorizerModel "
            f"is not supported in opset < {min_opset}"
        )

    # Create a TfIdfVectorizer node with gram length set to 1 and mode set to "TF".
    vectorizer_output_variable_name = scope.get_unique_variable_name(
        "vectorizer_output"
    )
    tfIdfVectorizer_attrs = {
        "name": scope.get_unique_operator_name("tfIdfVectorizer"),
        "min_gram_length": 1,
        "max_gram_length": 1,
        "max_skip_count": 0,
        "mode": "TF",
        "ngram_counts": [0],
        "ngram_indexes": [*range(len(vocab))],
        "pool_strings": vocab,
    }

    container.add_node(
        op_type="TfIdfVectorizer",
        inputs=[operator.inputs[0].full_name],
        outputs=[vectorizer_output_variable_name],
        op_version=9,
        **tfIdfVectorizer_attrs,
    )

    # In Spark's CountVectorizerModel, the comparison
    # with minTF is inclusive,
    # but in ThresholdedRelu (or Binarizer) node, the comparison
    # with `alpha` (or `threshold`) is exclusive.
    # So, we need to subtract epsilon from minTF to make the comparison
    # with `alpha` (or `threshold`) effectively inclusive.
    epsilon = 1e-6
    if binary:
        # Create a Binarizer node with threshold set to minTF - epsilon.
        container.add_node(
            op_type="Binarizer",
            inputs=[vectorizer_output_variable_name],
            outputs=[operator.outputs[0].full_name],
            op_version=1,
            op_domain="ai.onnx.ml",
            threshold=minTF - epsilon,
        )
    else:
        # Create a ThresholdedRelu node with alpha set to minTF - epsilon
        container.add_node(
            op_type="ThresholdedRelu",
            inputs=[vectorizer_output_variable_name],
            outputs=[operator.outputs[0].full_name],
            op_version=10,
            alpha=minTF - epsilon,
        )


register_converter("pyspark.ml.feature.CountVectorizerModel", convert_count_vectorizer)


def calculate_count_vectorizer_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[StringTensorType])

    N = operator.inputs[0].type.shape[0]
    C = len(operator.raw_operator.vocabulary)
    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator(
    "pyspark.ml.feature.CountVectorizerModel", calculate_count_vectorizer_output_shapes
)
