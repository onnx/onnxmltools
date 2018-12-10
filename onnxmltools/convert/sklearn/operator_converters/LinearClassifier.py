# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import six, numbers
from ...common._registration import register_converter
from ....proto import onnx_proto


def convert_sklearn_linear_classifier(scope, operator, container):
    op = operator.raw_operator
    coefficients = op.coef_.flatten().tolist()
    intercepts = op.intercept_.tolist()
    classes = op.classes_
    if len(classes) == 2:
        coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
        intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

    multi_class = 0
    if hasattr(op, 'multi_class'):
        if op.multi_class == 'ovr':
            multi_class = 1
        else:
            multi_class = 2

    classifier_type = 'LinearClassifier'
    classifier_attrs = {'name': scope.get_unique_operator_name(classifier_type)}

    # nb = NodeBuilder(context, 'LinearClassifier', op_domain='ai.onnx.ml')
    classifier_attrs['coefficients'] = coefficients
    classifier_attrs['intercepts'] = intercepts
    classifier_attrs['multi_class'] = 1 if multi_class == 2 else 0
    if op.__class__.__name__ == 'LinearSVC':
        classifier_attrs['post_transform'] = 'NONE'
    elif op.__class__.__name__ == 'LogisticRegression':
        if multi_class == 2:
            if len(op.classes_) == 2:
                # See method _predict_proba_lr.
                # When number if classes is two, the function
                # is not SOFTMAX.
                # https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/linear_model/base.py#L300
                classifier_attrs['post_transform'] = 'LOGISTIC'
            else:
                classifier_attrs['post_transform'] = 'LOGISTIC'
        else:
            classifier_attrs['post_transform'] = 'LOGISTIC'
    elif op.__class__.__name__ in ('SGDClassifier', 'LinearSVC'):
        classifier_attrs['post_transform'] = 'NONE'
    else:
        if multi_class == 2:
            classifier_attrs['post_transform'] = 'SOFTMAX'
        else:
            classifier_attrs['post_transform'] = 'LOGISTIC'

    if all(isinstance(i, (six.string_types, six.text_type)) for i in classes):
        class_labels = [str(i) for i in classes]
        classifier_attrs['classlabels_strings'] = class_labels
    elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in classes):
        class_labels = [int(i) for i in classes]
        classifier_attrs['classlabels_ints'] = class_labels
    else:
        raise RuntimeError('Label vector must be a string or a integer tensor')

    label_name = operator.outputs[0].full_name
    probability_tensor_name = scope.get_unique_variable_name('probability_tensor')
    
    if op.__class__.__name__ == 'LinearSVC' and op.classes_.shape[0] <= 2:
        raw_scores_tensor_name = scope.get_unique_variable_name('raw_scores_tensor')
        positive_class_index_name = scope.get_unique_variable_name('positive_class_index')

        container.add_initializer(positive_class_index_name, onnx_proto.TensorProto.INT64,
                              [], [1])

        container.add_node(classifier_type, operator.inputs[0].full_name,
                           [label_name, raw_scores_tensor_name],
                           op_domain='ai.onnx.ml', **classifier_attrs)
        container.add_node('ArrayFeatureExtractor', [raw_scores_tensor_name, positive_class_index_name],
                           operator.outputs[1].full_name, name=scope.get_unique_operator_name('ArrayFeatureExtractor'),
                           op_domain='ai.onnx.ml')
    else:
        container.add_node(classifier_type, operator.inputs[0].full_name,
                           [label_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **classifier_attrs)

        # Make sure the probability sum is 1 over all classes
        if multi_class > 0 and op.__class__.__name__ != 'LinearSVC':
            normalized_probability_tensor_name = scope.get_unique_variable_name(probability_tensor_name + '_normalized')
            normalizer_type = 'Normalizer'
            normalizer_attrs = {'name': scope.get_unique_operator_name(normalizer_type), 'norm': 'L1'}
            container.add_node(normalizer_type, probability_tensor_name, normalized_probability_tensor_name,
                               op_domain='ai.onnx.ml', **normalizer_attrs)
        else:
            normalized_probability_tensor_name = probability_tensor_name

        # Post-process probability tensor produced by LinearClassifier operator
        zipmap_type = 'ZipMap'
        zipmap_attrs = {'name': scope.get_unique_operator_name(zipmap_type)}
        if all(isinstance(i, (six.string_types, six.text_type)) for i in class_labels):
            zipmap_attrs['classlabels_strings'] = class_labels
        elif all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in class_labels):
            zipmap_attrs['classlabels_int64s'] = class_labels
        else:
            raise RuntimeError('Label vector must be a string or a integer tensor')

        container.add_node(zipmap_type, normalized_probability_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)


register_converter('SklearnLinearClassifier', convert_sklearn_linear_classifier)
register_converter('SklearnLinearSVC', convert_sklearn_linear_classifier)
