#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration


class MeanVarianceNormalizeLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'mvn')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        params = cm_node.mvn
        nb = NodeBuilder(context, 'MeanVarianceNormalization')
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)
        nb.add_attribute('across_channels', params.acrossChannels)
        nb.add_attribute('normalize_variance', params.normalizeVariance)

        return nb.make_node()


registration.register_nn_converter('mvn', MeanVarianceNormalizeLayerConverter)
