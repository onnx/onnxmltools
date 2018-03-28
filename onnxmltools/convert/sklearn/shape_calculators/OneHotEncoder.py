import collections, numbers
from ...coreml._data_types import FloatTensorType
from ...coreml.registration import register_shape_calculator


def calculate_sklearn_one_hot_encoder_output_shapes(operator):
    op = operator.raw_operator
    if op.categorical_features == 'all':
        C = operator.inputs[0].type.shape[1]
        categorical_feature_indices = [i for i in range(C)]
    elif isinstance(op.categorical_features, collections.Iterable):
        if all(isinstance(i, bool) for i in op.categorical_features):
            categorical_feature_indices = [i for i, active in enumerate(op.categorical_features) if active]
        else:
            categorical_feature_indices = [int(i) for i in op.categorical_features]
    else:
        raise ValueError('Unknown operation mode')

    # encoded_slot_sizes[i] is the number of output coordinates associated with the ith categorical feature
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
