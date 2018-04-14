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

    # Figure out which coordinates are categorical features we're going to encode
    if op.categorical_features == 'all':
        # In this case, all features need to be encoded
        C = operator.inputs[0].type.shape[1]
        categorical_feature_indices = [i for i in range(C)]
    elif isinstance(op.categorical_features, collections.Iterable):
        # In this case, there are two formats to specify which features are encoded.
        if all(isinstance(i, (bool, np.bool_)) for i in op.categorical_features):
            # op.categorical_features is a binary vector. Its ith element is 0/1 if the ith coordinate is not encoded/
            # encoded.
            categorical_feature_indices = [i for i, active in enumerate(op.categorical_features) if active]
        else:
            # op.categorical_features is a vector containing all categorical features' indexes.
            categorical_feature_indices = [int(i) for i in op.categorical_features]
    else:
        raise ValueError('Unknown operation mode')

    # Calculate the number of allowed categorical values in each original categorical coordinate.
    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature.
    encoded_slot_sizes = []
    if op.n_values == 'auto':
        # Use active feature to determine output length
        for i in range(len(op.feature_indices_) - 1):
            categorical_size = 0
            index_head = op.feature_indices_[i]
            index_tail = op.feature_indices_[i + 1]  # feature indexed by index_tail is not included in this category
            for j in op.active_features_:
                if index_head <= j and j < index_tail:
                    categorical_size += 1
            encoded_slot_sizes.append(categorical_size)
    elif isinstance(op.n_values, numbers.Integral):
        # Each categorical feature will be mapped to a fixed length one-hot sub-vector
        for i in range(len(op.feature_indices_) - 1):
            encoded_slot_sizes.append(op.n_values)
    else:
        # Each categorical feature has its own sub-vector length
        encoded_slot_sizes = [i for i in op.n_values]

    N = operator.inputs[0].type.shape[0]
    # Calculate the output feature length by replacing the count of categorical features with their encoded widths
    if operator.inputs[0].type.shape[1] != 'None':
        C = operator.inputs[0].type.shape[1] - len(categorical_feature_indices) + sum(encoded_slot_sizes)
    else:
        C = 'None'

    operator.outputs[0].type = FloatTensorType([N, C])


register_shape_calculator('SklearnOneHotEncoder', calculate_sklearn_one_hot_encoder_output_shapes)
