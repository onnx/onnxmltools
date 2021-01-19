# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_mul
from ....common._registration import register_converter


def convert_dot(scope, operator, container):
    # To calculate cosine similarity, we first use LpNormalization to make the two input vectors unit-length.
    # Then, we calculate element-wise product of the two unit-length vectors. Finally, the similarity is the
    # sum of all the product's elements. Notice that we carefully specify the axis of the subsequent operators,
    # so they can work properly with a batch of vectors.

    if operator.raw_operator.dot.cosineSimilarity:
        # Normalize the first input and store the result on a temporal variable
        intra_variable_name1 = scope.get_unique_variable_name(operator.inputs[0].full_name + '_normalized')
        normalizer_name1 = scope.get_unique_operator_name('L2NormNormalizer')
        attrs1 = {'name': normalizer_name1, 'p': 2, 'aixs': 1}
        container.add_node('LpNormalization', [operator.inputs[0].full_name], [intra_variable_name1], **attrs1)

        # Normalize the second input and store the result on a temporal variable
        intra_variable_name2 = scope.get_unique_variable_name(operator.inputs[1].full_name + '_normalized')
        normalizer_name2 = scope.get_unique_operator_name('L2NormNormalizer')
        attrs2 = {'name': normalizer_name2, 'p': 2, 'aixs': 1}
        container.add_node('LpNormalization', [operator.inputs[1].full_name], [intra_variable_name2], **attrs2)
    else:
        # This case is a simple dot product; no normalization is required.
        intra_variable_name1 = operator.inputs[0].full_name
        intra_variable_name2 = operator.inputs[1].full_name

    # Do element-wise product of the two unit-length tensors
    product_name = scope.get_unique_variable_name(intra_variable_name1 + '_multiply_' + intra_variable_name2)
    apply_mul(scope, [intra_variable_name1, intra_variable_name2], product_name, container, broadcast=0)

    # Sum up results from different dimensions to get the final cosine similarity
    reducer_name = scope.get_unique_operator_name('ReduceSum')
    reducer_attrs = {'name': reducer_name, 'axes': [1], 'keepdims': 1}
    container.add_node('ReduceSum', [product_name], operator.output_full_names, **reducer_attrs)


register_converter('dot', convert_dot)
