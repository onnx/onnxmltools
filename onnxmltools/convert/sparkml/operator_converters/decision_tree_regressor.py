# SPDX-License-Identifier: Apache-2.0

from ...common.data_types import FloatTensorType
from ...common.tree_ensemble import add_tree_to_attribute_pairs, \
    get_default_tree_regressor_attribute_pairs
from ...common.utils import check_input_and_output_numbers
from ...sparkml.operator_converters.decision_tree_classifier import save_read_sparkml_model_data
from ...sparkml.operator_converters.tree_ensemble_common import sparkml_tree_dataset_to_sklearn
from ...common._registration import register_converter, register_shape_calculator


def convert_decision_tree_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs['n_targets'] = 1

    tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], op)
    tree = sparkml_tree_dataset_to_sklearn(tree_df, is_classifier=False)
    add_tree_to_attribute_pairs(attrs, False, tree, 0, 1., 0, False)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names,
                       op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.regression.DecisionTreeRegressionModel', convert_decision_tree_regressor)


def calculate_decision_tree_regressor_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType(shape=[N, 1])


register_shape_calculator('pyspark.ml.regression.DecisionTreeRegressionModel',
                          calculate_decision_tree_regressor_output_shapes)
