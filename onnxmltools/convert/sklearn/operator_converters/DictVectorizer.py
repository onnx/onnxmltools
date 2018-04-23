# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers, six
from ...common._registration import register_converter


def convert_sklearn_dict_vectorizer(scope, operator, container):
    op_type = 'DictVectorizer'
    op = operator.raw_operator
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    if all(isinstance(feature_name, (six.string_types, six.text_type)) for feature_name in op.feature_names_):
        attrs['string_vocabulary'] = list(str(i) for i in op.feature_names_)
    elif all(isinstance(feature_name, numbers.Integral) for feature_name in op.feature_names_):
        attrs['int64_vocabulary'] = list(int(i) for i in op.feature_names_)
    else:
        raise ValueError('Unsupported key type found')

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnDictVectorizer', convert_sklearn_dict_vectorizer)
