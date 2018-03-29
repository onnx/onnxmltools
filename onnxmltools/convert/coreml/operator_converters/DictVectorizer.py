# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter


def convert_dictionary_vectorizer(scope, operator, container):
    op_type = 'DictVectorizer'
    attrs = {'name': operator.full_name}
    raw_model = operator.raw_operator.dictVectorizer
    if raw_model.HasField('stringToIndex'):
        attrs['string_vocabulary'] = raw_model.stringToIndex.vector
    else:
        attrs['int64_vocabulary'] = raw_model.int64ToIndex.vector

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name],
                       op_domain='ai.onnx.ml', **attrs)


register_converter('dictVectorizer', convert_dictionary_vectorizer)
