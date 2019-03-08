# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from onnxmltools.convert.common.tree_ensemble import get_default_tree_classifier_attribute_pairs, \
    add_tree_to_attribute_pairs
from ...common._registration import register_converter, register_shape_calculator

def convert_tree_ensemble_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs["classlabels_int64s"] = range(0, op.numClasses)

    tree_df = save_read_sparkml_model_data(operator.raw_operator_params['SparkSession'], op)
    tree = sparkml_tree_dataset_to_sklearn(tree_df)
    add_tree_to_attribute_pairs(attrs, True, tree, 0, 1., 0, True)

    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name,
                       operator.outputs[1].full_name], op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.DecisionTreeClassificationModel', convert_tree_ensemble_classifier)


def calculate_tree_ensemble_classifier_output_shapes(operator):


register_shape_calculator('pyspark.ml.classification.DecisionTreeClassificationModel', convert_tree_ensemble_classifier)


class SparkMLTree(dict):
    pass


def sparkml_tree_dataset_to_sklearn(tree_df):
    feature = []
    threshold = []
    tree_pandas = tree_df.toPandas()
    children_left = tree_pandas.leftChild.values.tolist()
    children_right = tree_pandas.rightChild.values.tolist()
    value = tree_pandas.impurityStats.values.tolist()
    split = tree_pandas.split.values
    for i, item in enumerate(split):
        feature[i] = item[0]
        threshold[i] = item[1][0] if len(item[1]) >= 1 else -1.0
    tree = SparkMLTree()
    tree.children_left = children_left
    tree.children_right = children_right
    tree.value = value
    tree.feature = feature
    tree.threshold = threshold
    return tree

def save_read_sparkml_model_data(spark, model):
    import tempfile
    import os
    path = os.path.join(tempfile.tempdir, type(x).__name__)
    model.write().overwrite().save(path)
    df = spark.read.parquet(os.path.join(path, 'data'))
    return df
