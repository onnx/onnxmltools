# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import numbers, six
from ...common._registration import register_converter


def convert_sklearn_svm(scope, operator, container):
    svm_attrs = {'name': scope.get_unique_operator_name('SVM')}
    op = operator.raw_operator
    if isinstance(op.dual_coef_, np.ndarray):
        coef = op.dual_coef_.ravel().tolist()
    else:
        coef = op.dual_coef_
    intercept = op.intercept_
    if isinstance(op.support_vectors_, np.ndarray):
        support_vectors = op.support_vectors_.ravel().tolist()
    else:
        support_vectors = op.support_vectors_

    svm_attrs['kernel_type'] = op.kernel.upper()
    svm_attrs['kernel_params'] = [float(_) for _ in [op._gamma, op.coef0, op.degree]]
    svm_attrs['support_vectors'] = support_vectors

    if operator.type in ['SklearnSVC', 'SklearnNuSVC'] and len(op.classes_) == 2:
        svm_attrs['coefficients'] = [-v for v in coef]
        svm_attrs['rho'] = [-v for v in intercept]
    else:
        svm_attrs['coefficients'] = coef
        svm_attrs['rho'] = intercept

    if operator.type in ['SklearnSVC', 'SklearnNuSVC']:
        op_type = 'SVMClassifier'

        if len(op.probA_) > 0:
            svm_attrs['prob_a'] = op.probA_
        if len(op.probB_) > 0:
            svm_attrs['prob_b'] = op.probB_

        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['vectors_per_class'] = op.n_support_.tolist()

        label_name = operator.outputs[0].full_name
        probability_tensor_name = scope.get_unique_variable_name('probability_tensor')

        zipmap_attrs = {'name': scope.get_unique_operator_name('ZipMap')}
        if all(isinstance(i, (numbers.Real, bool, np.bool_)) for i in op.classes_):
            labels = [int(i) for i in op.classes_]
            svm_attrs['classlabels_ints'] = labels
            zipmap_attrs['classlabels_int64s'] = labels
        elif all(isinstance(i, (six.text_type, six.string_types)) for i in op.classes_):
            labels = [str(i) for i in op.classes_]
            svm_attrs['classlabels_strings'] = labels
            zipmap_attrs['classlabels_strings'] = labels
        else:
            raise RuntimeError('Invalid class label type [%s]' % op.classes_)

        container.add_node(op_type, operator.inputs[0].full_name, [label_name, probability_tensor_name],
                           op_domain='ai.onnx.ml', **svm_attrs)
        container.add_node('ZipMap', probability_tensor_name, operator.outputs[1].full_name,
                           op_domain='ai.onnx.ml', **zipmap_attrs)

    elif operator.type in ['SklearnSVR', 'SklearnNuSVR']:
        op_type = 'SVMRegressor'
        svm_attrs['post_transform'] = 'NONE'
        svm_attrs['n_supports'] = len(op.support_)

        container.add_node(op_type, operator.input_full_names, operator.output_full_names,
                           op_domain='ai.onnx.ml', **svm_attrs)
    else:
        raise ValueError('Unknown support vector machien model type found')


register_converter('SklearnSVC', convert_sklearn_svm)
register_converter('SklearnSVR', convert_sklearn_svm)
