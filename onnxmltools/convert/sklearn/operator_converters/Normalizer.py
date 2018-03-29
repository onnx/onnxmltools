# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter
from .common import concatenate_variables


def convert_sklearn_normalizer(scope, operator, container):
    if len(operator.inputs) > 1:
        # If there are multiple input tensors, we combine them using a FeatureVectorizer
        feature_name = concatenate_variables(scope, operator.inputs, container)
    else:
        # No concatenation is needed, we just use the first variable's name
        feature_name = operator.inputs[0].full_name

    op = operator.raw_operator
    op_type = 'Normalizer'
    norm_map = {'max': 'MAX', 'l1': 'L1', 'l2': 'L2'}
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    if op.norm in norm_map:
        attrs['norm'] = norm_map[op.norm]
    else:
        raise RuntimeError('Invalid norm: %s' % op.norm)

    container.add_node(op_type, feature_name, operator.outputs[0].full_name, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnNormalizer', convert_sklearn_normalizer)
