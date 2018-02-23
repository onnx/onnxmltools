#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d


class MeanImagePreprocessorConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'meanImage')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network preprocessing')

    @staticmethod
    def convert(context, cm_node, input, output):
        extend_inputs_from_2d_to_4d(context, input)

        nb = NodeBuilder(context, 'MeanSubtraction')
        nb.add_attribute('image', cm_node.meanImage)
        nb.add_input(input)
        nb.add_output(output)

        return nb.make_node()


registration.register_nn_converter('meanImage', MeanImagePreprocessorConverter)
