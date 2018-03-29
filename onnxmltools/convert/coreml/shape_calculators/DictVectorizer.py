# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._data_types import DictionaryType, SequenceType, FloatTensorType
from ...common._registration import register_shape_calculator


def calculate_dictionary_vectorizer_output_shapes(operator):
    # We assume all dictionaries' value types are float. It seems be reasonable to CoreML's
    # model input, but the existence of other map types leads to some concerns.
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Dictionary vectorizer operator has only one input and output')

    # [TODO] dictionary vectorizer should be able to accept a sequence of dictionary
    if type(operator.inputs[0].type) != DictionaryType and type(operator.inputs[0].type) != SequenceType:
        raise RuntimeError('Input type must be a sequence of dictionary or a dictionary of a sequence')

    params = operator.raw_operator.dictVectorizer
    string_key_vector = params.stringToIndex.vector
    int64_key_vector = params.int64ToIndex.vector

    if len(string_key_vector) > 0 and len(int64_key_vector) > 0:
        raise RuntimeError('Only one key type can present at the same time')

    doc_string = operator.outputs[0].type.doc_string
    if len(string_key_vector) > 0:
        operator.outputs[0].type = FloatTensorType([1, len(string_key_vector)], doc_string=doc_string)
    elif len(int64_key_vector) > 0:
        operator.outputs[1].type.shape = FloatTensorType([1, len(int64_key_vector)], doc_string=doc_string)
    else:
        raise ValueError('Key vector cannot be empty')


register_shape_calculator('dictVectorizer', calculate_dictionary_vectorizer_output_shapes)
