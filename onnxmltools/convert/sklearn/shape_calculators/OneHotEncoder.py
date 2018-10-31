# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import collections, numbers
from ...common._registration import register_shape_calculator
from ...common.data_types import FloatTensorType


def calculate_sklearn_one_hot_encoder_output_shapes(operator):
    '''
    Scikit-learn's one-hot encoder maps (some selected) categorical values into some binary sub-vectors. For each
    sub-vector, all coordinates but one are zeros. For example, if ['a', 'b', 1.] and ['c', 'c', 2.] are used to train a
    scikit-learn one-hot encoder and only the first two coordinates are encoded. Feeding ['a', 'b', 1.] into the trained
    encoder may look like
    >>> [1., 0.] + [1., 0.] + [1.]
    There are three sub-vectors. The first/second/third one is a binary representation of ['a']/['b']/[1.]. Note that
    we use Python list for the sake of simplicity. The actual output type is either numpy or scipy array. Similarly, the
    outcome of ['c', 'c', 2.] may be
    >>> [0., 1.] + [0., 1.] + [2.]

    Calculating non-encoded coordinates' length is simple because scikit-learn just append them after encoded features.
    For categorical values, we need to figure out which coordinates are being encoded and how many categorical values
    are allowed in each coordinate. In our example, there are two/two categorical values in the first/second coordinate
    so the total lenght of encoded vector is
    >>> 2 + 2 + 1  # two categorical values + two categorical values + one non-encoded feature

    Allowed input/output patterns are
        1. [N, C] ---> [N, C']
        2. [N, 'None'] ---> [N, 'None']
    '''
    op = operator.raw_operator

    categorical_feature_indices = [i for i, mat in enumerate(op.categories_) if mat is not None and len(mat) > 0]

    # Calculate the number of allowed categorical values in each original categorical coordinate.
    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature.
    encoded_slot_sizes = []

    # Use active feature to determine output length
    index_head = 0
    for i in range(len(op.categories_)):
        if op.categories_[i] is None or len(op.categories_[i]) == 0:
            continue
        categorical_size = op.categories_[i].shape[0]
        # feature indexed by index_tail is not included in this category
        index_tail = index_head + categorical_size
        encoded_slot_sizes.append(categorical_size)

    N = operator.inputs[0].type.shape[0]
    # Calculate the output feature length by replacing the count of categorical
    # features with their encoded widths
    if operator.inputs[0].type.shape[1] != 'None':
        C = operator.inputs[0].type.shape[1] - len(categorical_feature_indices) + sum(encoded_slot_sizes)
    else:
        C = 'None'

    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('SklearnOneHotEncoder', calculate_sklearn_one_hot_encoder_output_shapes)
