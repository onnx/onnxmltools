# SPDX-License-Identifier: Apache-2.0

import logging
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator
from .tree_ensemble_common import (
    save_read_sparkml_model_data,
    sparkml_tree_dataset_to_sklearn,
    add_tree_to_attribute_pairs,
    get_default_tree_classifier_attribute_pairs,
)
from .tree_helper import rewrite_ids_and_process

logger = logging.getLogger("onnxmltools")


def convert_decision_tree_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = "TreeEnsembleClassifier"

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs["name"] = scope.get_unique_operator_name(op_type)
    attrs["classlabels_int64s"] = list(range(0, op.numClasses))

    logger.info("[convert_decision_tree_classifier] save_read_sparkml_model_data")
    tree_df = save_read_sparkml_model_data(operator.raw_params["SparkSession"], op)
    logger.info("[convert_decision_tree_classifier] sparkml_tree_dataset_to_sklearn")
    tree = sparkml_tree_dataset_to_sklearn(tree_df, is_classifier=True)
    logger.info("[convert_decision_tree_classifier] add_tree_to_attribute_pairs")
    add_tree_to_attribute_pairs(
        attrs, True, tree, 0, 1.0, 0, leaf_weights_are_counts=True
    )
    logger.info(
        "[convert_decision_tree_classifier] n_nodes=%d", len(attrs["nodes_nodeids"])
    )

    # Some values appear in an array of one element instead of a float.

    new_attrs = rewrite_ids_and_process(attrs, logger)

    container.add_node(
        op_type,
        operator.input_full_names,
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain="ai.onnx.ml",
        **new_attrs
    )


register_converter(
    "pyspark.ml.classification.DecisionTreeClassificationModel",
    convert_decision_tree_classifier,
)


def calculate_decision_tree_classifier_output_shapes(operator):
    check_input_and_output_numbers(
        operator, input_count_range=1, output_count_range=[1, 2]
    )
    check_input_and_output_types(
        operator, good_input_types=[FloatTensorType, Int64TensorType]
    )
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Input must be a [N, C]-tensor")

    N = operator.inputs[0].type.shape[0]

    class_count = operator.raw_operator.numClasses
    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator(
    "pyspark.ml.classification.DecisionTreeClassificationModel",
    calculate_decision_tree_classifier_output_shapes,
)
