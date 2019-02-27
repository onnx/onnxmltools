# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter, register_shape_calculator

def convert_tree_ensemble_classifier(scope, operator, container):
    op = operator.raw_operator
    op_type = 'TreeEnsembleClassifier'

    attrs = get_default_tree_classifier_attribute_pairs()
    attrs['name'] = scope.get_unique_operator_name(op_type)
    attrs["classlabels_int64s"] = range(0, op.numClasses)
:
    add_tree_to_attribute_ptest_linear_regressor.pyairs(attrs, True, op.tree_, 0, 1., 0, True)

    container.add_node(op_type, operator.input_full_names, [operator.outputs[0].full_name,
                       operator.outputs[1].full_name], op_domain='ai.onnx.ml', **attrs)


register_converter('pyspark.ml.classification.DecisionTreeClassificationModel', convert_tree_ensemble_classifier)


def calculate_tree_ensemble_classifier_output_shapes(operator):


register_shape_calculator('pyspark.ml.classification.DecisionTreeClassificationModel', convert_tree_ensemble_classifier)


def get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['class_treeids'] = []
    attrs['class_nodeids'] = []
    attrs['class_ids'] = []
    attrs['class_weights'] = []
    return attrs

