#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sklearn
from ..common import register_converter
from ..common import utils
from ..common import model_util


class OneHotEncoderConverter:

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'feature_indices_')
            utils._check_has_attr(sk_node, 'active_features_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        feature_indices = sk_node.feature_indices_
        active_features = sk_node.active_features_
        categorical_features = sk_node.categorical_features
        feature_vector_inputs = []
        node_chain = []
        initial_val = 0
        final_val = 0
        ohe_idx = 0

        # Calculate the number of inputs
        assert (len(inputs) == 1)
        num_features = model_util.get_feature_count(inputs[0])
        range_features = range(num_features)
        categorical_indices = list(range_features)
        if categorical_features != 'all':
            categorical_indices = categorical_features

        for idx in range_features:
            if idx not in categorical_indices:
                continue

            # Only create an OHE for the categorical indices specified
            base = feature_indices[ohe_idx]
            max_val = feature_indices[ohe_idx + 1]

            for initial_val in range(final_val, len(active_features)):
                if active_features[initial_val] >= base:
                    break

            for final_val in range(len(active_features) - 1, initial_val, -1):
                if active_features[final_val] <= max_val:
                    if active_features[final_val] < max_val or final_val == len(active_features) - 1:
                        final_val += 1
                    break

            categories = active_features[initial_val:final_val] - [base] * (final_val - initial_val)
            feature_extractor = model_util.create_feature_extractor(inputs[0], 'ohe_fe_output', [idx], context)
            node_chain.append(feature_extractor)
            ohe_node = model_util.create_ohe(feature_extractor.outputs[0], 'ohe_output', categories, context)
            node_chain.append(ohe_node)
            feature_vector_inputs.append(ohe_node.outputs[0])

            ohe_idx += 1

        # Handle non-encoded values. These are added to the end via the feature vectorizer.
        indices = [x for x in range_features if x not in categorical_indices]
        if len(indices) > 0:
            feature_extractor_pass = model_util.create_feature_extractor(inputs[0], 'ohe_pass_output', indices, context)
            node_chain.append(feature_extractor_pass)
            feature_vector_inputs.append(feature_extractor_pass.outputs[0])

        # Create a feature vectorizer to join the results.
        feature_vector = model_util.create_feature_vector(feature_vector_inputs, 'ohe_fv_output', context)
        node_chain.append(feature_vector)

        return node_chain


# Register the class for processing
register_converter(sklearn.preprocessing.OneHotEncoder, OneHotEncoderConverter)
