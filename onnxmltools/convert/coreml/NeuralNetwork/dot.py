#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration


class DotProductLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'dot')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        if cm_node.dot.cosineSimilarity:
            # To calculate cosine similarity, we first use LpNormalization to make the two input vectors unit-length.
            # Then, we calculate element-wise product of the two unit-length vectors. Finally, the similarity is the
            # sum of all the product's elements. Notice that we carefully specify the axis of the subsequent operators,
            # so they can work properly with a batch of vectors.

            # Normalize the first input and store the result on a temporal variable
            nb1 = NodeBuilder(context, 'LpNormalization')
            nb1.extend_input(inputs[0])
            nb1.add_attribute('p', 2)
            nb1.add_attribute('axis', 1)
            nb1.add_output(nb1.name)

            # Normalize the second input and store the result on a temporal variable
            nb2 = NodeBuilder(context, 'LpNormalization')
            nb2.extend_inputs(inputs[1])
            nb2.add_attribute('p', 2)
            nb2.add_attribute('axis', 1)
            nb2.add_output(nb2.name)

            # Do element-wise product of the two unit-length tensors
            nb3 = NodeBuilder(context, 'Mul')
            nb3.extend_inputs(nb1.output_names)
            nb3.extend_inputs(nb2.output_names)
            nb3.add_output(nb3.name)

            # Sum up results from different dimensions to get the final cosine similarity
            nb4 = NodeBuilder(context, 'ReduceSum')
            nb4.extend_inputs(nb3.output_names)
            nb4.extend_outputs(outputs)
            nb4.add_attribute('axes', [1])
            nb4.add_attribute('keepdims', False)

            return [nb.make_node() for nb in [nb1, nb2, nb3, nb4]]
        else:
            # This case is a simple dot product, which can be formed by a element-wise multiplication followed by
            # a reduction.

            # Calculate the element-wise product of inputs
            nb1 = NodeBuilder(context, 'Mul')
            nb1.extend_inputs(inputs)
            nb1.add_output(nb1.name)

            # Aggregate the product across all coordinates
            nb2 = NodeBuilder(context, 'ReduceSum')
            nb2.extend_inputs(nb1.output_names)
            nb2.extend_outputs(outputs)
            nb2.add_attribute('axes', [1])
            nb2.add_attribute('keepdims', False)

            return [nb.make_node() for nb in [nb1, nb2]]


registration.register_nn_converter('dot', DotProductLayerConverter)
