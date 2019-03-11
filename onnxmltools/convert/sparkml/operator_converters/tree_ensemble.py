# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnxmltools.convert.common.data_types import Int64TensorType, FloatTensorType
from onnxmltools.convert.common.tree_ensemble import get_default_tree_classifier_attribute_pairs, \
    add_tree_to_attribute_pairs, get_default_tree_regressor_attribute_pairs
from onnxmltools.convert.common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


###
### Classifier Secion
###
def convert_tree_ensemble_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs["classlabels_int64s"] = range(0, op.numClasses)

    tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], op)
    tree = sparkml_tree_dataset_to_sklearn(tree_df)
    add_tree_to_attribute_pairs(attrs, True, tree, 0, 1., 0, True)

    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name,
                       operator.outputs[1].full_name], op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.DecisionTreeClassificationModel', convert_tree_ensemble_classifier)


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
        tree = sparkml_tree_dataset_to_sklearn(tree_df)
        add_tree_to_attribute_pairs(attr_pairs, True, tree, tree_id,
                                    # op.treeWeights[tree_id],
                                    tree_weight, 0, True)

    container.add_node(
        op_type, operator.input_full_names,
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain='ai.onnx.ml', **attr_pairs)


register_converter('pyspark.ml.classification.RandomForestClassificationModel', convert_random_forest_classifier)


def calculate_tree_ensemble_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]

    class_count = operator.raw_operator.numClasses
    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator('pyspark.ml.classification.DecisionTreeClassificationModel',
                          calculate_tree_ensemble_classifier_output_shapes)
register_shape_calculator('pyspark.ml.classification.RandomForestClassificationModel',
                          calculate_tree_ensemble_classifier_output_shapes)

###
### Regressor Secion
###
def convert_tree_ensemble_regressor(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleRegressor'

    attrs = get_default_tree_regressor_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    ## need to investigate the following
    #attrs['n_targets'] = int(op.n_outputs_)

    tree_df = save_read_sparkml_model_data(operator.raw_params['SparkSession'], op)
    tree = sparkml_tree_dataset_to_sklearn(tree_df)
    add_tree_to_attribute_pairs(attrs, True, tree, 0, 1., 0, True)

    container.add_node(op_type, operator.input_full_names, operator.outputs[0].output_full_names,
                       op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.DecisionTreeRegressionModel', convert_tree_ensemble_regressor)


def calculate_tree_ensemble_regressor_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = FloatTensorType(shape=[N,1])


register_shape_calculator('pyspark.ml.classification.DecisionTreeRegressionModel', calculate_tree_ensemble_regressor_output_shapes)


###
### Common Secion
###
class SparkMLTree(dict):
    pass


def sparkml_tree_dataset_to_sklearn(tree_df):
    import numpy
    feature = []
    threshold = []
    tree_pandas = tree_df.toPandas()
    children_left = tree_pandas.leftChild.values.tolist()
    children_right = tree_pandas.rightChild.values.tolist()
    value = tree_pandas.impurityStats.values.tolist()
    split = tree_pandas.split.apply(tuple).values
    for item in split:
        feature.append(item[0])
        threshold.append(item[1][0] if len(item[1]) >= 1 else -1.0)
    tree = SparkMLTree()
    tree.children_left = children_left
    tree.children_right = children_right
    tree.value = numpy.asarray(value, dtype=numpy.float32)
    tree.feature = feature
    tree.threshold = threshold
    tree.node_count = tree_df.count()
    return tree


def save_read_sparkml_model_data(spark, model):
    import tempfile
    import os
    path = os.path.join(tempfile.tempdir, type(model).__name__)
    model.write().overwrite().save(path)
    df = spark.read.parquet(os.path.join(path, 'data'))
    #df.show(100)
    return df
