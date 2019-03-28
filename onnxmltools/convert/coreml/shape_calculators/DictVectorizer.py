# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_shape_calculator
from ...common.data_types import DictionaryType, SequenceType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_dictionary_vectorizer_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. Map ---> [1, C]

    C is the number of all allowed keys in the input dictionary.
    '''
    # We assume all dictionaries' value types are float. It seems be reasonable to CoreML's
    # model input, but the existence of other map types leads to some concerns.
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    # Two types are allowed. One is DictionaryType and the other one, SequenceType, means a sequence of dictionaries.
    check_input_and_output_types(operator, good_input_types=[DictionaryType, SequenceType])

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
