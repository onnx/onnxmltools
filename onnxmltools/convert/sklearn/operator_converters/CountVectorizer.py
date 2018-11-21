# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers, six
from ...common._registration import register_converter


def convert_sklearn_count_vectorizer(scope, operator, container):
        
    op = operator.raw_operator
    if op.analyzer != "word":
        raise NotImpplementedError("CountVectorizer cannot be converted, only tokenizer='word' is supported.")
    if op.strip_accents is not None:
        raise NotImpplementedError("CountVectorizer cannot be converted, only stip_accents=None is supported.")
        
    if op.lowercase or op.stop_words_:
        # StringNormalizer
        
        op_type = 'StringNormalizer'
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        attrs.update({
            'casechangeaction': 'LOWER',
            'stopwords': op.stop_words_,
            'iscasesenstive': not op.lowercase,
        })
        normalized = scope.get_unique_variable_name('normalized')
        container.add_node(op_type, operator.input_full_names, 
                           normalized, op_domain='ai.onnx.ml', **attrs)
    else:
        normalized = operator.input_full_names
        
    # Tokenizer
    padvalue = "#"
    while padvalue in op.vocabulary_:
        padvalue += "#"
    
    op_type = 'WordTokenizer'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs.update({
        'padvalue': padvalue,
        'separators': [' '],
    })

    tokenized = scope.get_unique_variable_name('tokenized')
    container.add_node(op_type, normalized, tokenized,
                       op_domain='ai.onnx.ml', **attrs)

    # Ngram    
    C = max(op.vocabulary_.values()) + 1
    words = [None for i in range(C)]
    weights = [0 for i in range(C)]
    indices = [None for i in range(C)]
    for k, v in op.vocabulary_.items():
        words[v] = k
        weights[v] = 1.
        
    # ngramcounts: list of int64s (type: AttributeProto::INTS). The starting indexes of 
    # 1-grams, 2-grams, and so on in pool. It is useful when determining the boundary 
    # between two consecutive collections of n-grams. For example, if ngramcounts 
    # is [0, 17, 36], the first index (zero-based) of 1-gram/2-gram/3-gram 
    # in pool are 0/17/36.
    # Scikit-learn sorts n-grams by alphabetical order and not by n
    split_words = [w.split() for w in words]
    ngcounts = [0]
    for i in range(1, len(split_words)):
        if len(split_words[i-1]) != len(split_words[i]):
            ngcounts.append(i)
    
    op_type = 'Ngram'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs.update({
        'M': op.ngram_range[0],
        'N': op.ngram_range[1],
        'mode': 'TF',
        'S': 0,
        'all': True,  # is this really useful is M is specified?
        'pool_strings': words,
        'pool_int64s': [],
        'ngramcounts': ngcounts,
        'weights': weights,
    })

    container.add_node(op_type, tokenized, operator.output_full_names,
                       op_domain='ai.onnx.ml', **attrs)


register_converter('SklearnCountVectorizer', convert_sklearn_count_vectorizer)

