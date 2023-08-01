# SPDX-License-Identifier: Apache-2.0

from ...common.tree_ensemble import get_default_tree_classifier_attribute_pairs, \
    add_tree_to_attribute_pairs
from ...common._registration import register_converter, register_shape_calculator
from .tree_ensemble_common import save_read_sparkml_model_data, sparkml_tree_dataset_to_sklearn
from .decision_tree_classifier import calculate_decision_tree_classifier_output_shapes


def convert_random_forest_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attr_pairs = get_default_tree_classifier_attribute_pairs()
    attr_pairs['name'] = scope.get_unique_operator_name(op_type)
    attr_pairs['classlabels_int64s'] = range(0, op.numClasses)

    # random forest calculate the final score by averaging over all trees'
    # outcomes, so all trees' weights are identical.
    tree_weight = 1. / op.getNumTrees

    for tree_id in range(0, op.getNumTrees):
        tree_model = op.trees[tree_id]
        tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], tree_model)
        tree = sparkml_tree_dataset_to_sklearn(tree_df, is_classifier=True)
        add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id,
                                    tree_weight, 0, True)

    container.add_node(
        op_type, operator.input_full_names,
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain='ai.onnx.ml', **attr_pairs)


register_converter('pyspark.ml.classification.RandomForestClassificationModel', convert_random_forest_classifier)

register_shape_calculator('pyspark.ml.classification.RandomForestClassificationModel',
                          calculate_decision_tree_classifier_output_shapes)
