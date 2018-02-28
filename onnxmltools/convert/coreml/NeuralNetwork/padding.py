#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d


class PaddingLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'padding')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.padding

        nb = NodeBuilder(context, 'Pad', op_version=2)

        pad_table = {'constant': 'constant',
                     'reflection': 'reflect',
                     'replication': 'edge'}

        pad_type = params.WhichOneof('PaddingType')
        if pad_type not in pad_table:
            raise ValueError('Unsupported padding mode: {}'.format(pad_type))
        nb.add_attribute('mode', pad_table[pad_type])

        # CoreML only pads for their H- and W- axes. Here we assume the shape of the tensor to be padded
        # is [N, C, H, W], so we have 8 padding amounts
        #     pads = [N_begin_index, C_begin_index, H_begin_index, W_begin_index,
        #             N_end_index,   C_end_index,   H_end_index,   W_end_index]
        # Because only H- and W-axes are padded in CoreML, we leave padding amounts of N- and C-axes zeros.
        pads = [0, 0, 0, 0, 0, 0, 0, 0]
        if len(params.paddingAmounts.borderAmounts) > 0:
            # Set H_begin_index
            pads[2] = params.paddingAmounts.borderAmounts[0].startEdgeSize
            # Set W_begin_index
            pads[3] = params.paddingAmounts.borderAmounts[1].startEdgeSize
            # Set H_end_index
            pads[6] = params.paddingAmounts.borderAmounts[0].endEdgeSize
            # Set W_end_index
            pads[7] = params.paddingAmounts.borderAmounts[1].endEdgeSize
        nb.add_attribute('pads', pads)

        if pad_type == 'constant':
            nb.add_attribute('value', params.constant.value)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


registration.register_nn_converter('padding', PaddingLayerConverter)
