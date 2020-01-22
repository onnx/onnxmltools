# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from ...common.tree_ensemble import add_tree_to_attribute_pairs, \
    get_default_tree_regressor_attribute_pairs
from ...common._registration import register_converter, register_shape_calculator
from .decision_tree_classifier import save_read_sparkml_model_data
from .decision_tree_regressor import calculate_decision_tree_regressor_output_shapes
from .tree_ensemble_common import sparkml_tree_dataset_to_sklearn


def convert_random_forest_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1

    # random forest calculate the final score by averaging over all trees'
    # outcomes, so all trees' weights are identical.
    tree_weight = 1. / op.getNumTrees

    for tree_id in range(0, op.getNumTrees):
        tree_model = op.trees[tree_id]
        tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], tree_model)
        tree = sparkml_tree_dataset_to_sklearn(tree_df, is_classifier=False)
        add_tree_to_attribute_pairs(attrs, False, tree, tree_id,
                                    tree_weight, 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names[0],
                       op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.regression.RandomForestRegressionModel', convert_random_forest_regressor)


register_shape_calculator('pyspark.ml.regression.RandomForestRegressionModel',
                          calculate_decision_tree_regressor_output_shapes)
