#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d


class LRNLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'lrn')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.lrn
        nb = NodeBuilder(context, 'LRN')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        nb.add_attribute('size', params.localSize)
        nb.add_attribute('alpha', params.alpha)
        nb.add_attribute('beta', params.beta)
        nb.add_attribute('bias', params.k if params.k != 0 else 1.)

        return nb.make_node()


registration.register_nn_converter('lrn', LRNLayerConverter)
