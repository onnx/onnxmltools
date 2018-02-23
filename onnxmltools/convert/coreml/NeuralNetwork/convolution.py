#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import model_util
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import SamePadding


class ConvolutionLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'convolution')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.convolution
        op_type = 'Conv' if not params.isDeconvolution else 'ConvTranspose'
        nb = NodeBuilder(context, op_type)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        groups = 1
        if params.nGroups != 0:
            groups = params.nGroups
        nb.add_attribute('group', groups)

        dims = [params.outputChannels, params.kernelChannels, params.kernelSize[0], params.kernelSize[1]]
        if params.isDeconvolution:
            dims[0] = params.kernelChannels
            dims[1] = int(params.outputChannels / groups)
        tensor_w = model_util.make_tensor('W', onnx_proto.TensorProto.FLOAT, dims, params.weights.floatValue)
        nb.add_initializer(tensor_w)

        if params.hasBias:
            dims = [len(params.bias.floatValue)]
            tensor_b = model_util.make_tensor('b', onnx_proto.TensorProto.FLOAT, dims, params.bias.floatValue)
            nb.add_initializer(tensor_b)

        dilations = [1, 1]
        if len(params.dilationFactor) > 0:
            dilations = [params.dilationFactor[0], params.dilationFactor[1]]
        nb.add_attribute('dilations', dilations)

        kernel_shape = [3, 3]
        if len(params.kernelSize) > 0:
            kernel_shape = params.kernelSize
        nb.add_attribute('kernel_shape', kernel_shape)

        pads = None
        auto_pad = None
        pad_type = params.WhichOneof('ConvolutionPaddingType')
        if pad_type == 'valid':

            if len(params.valid.paddingAmounts.borderAmounts) > 0:
                pads = [0, 0, 0, 0]
                pads[0] = params.valid.paddingAmounts.borderAmounts[0].startEdgeSize
                pads[1] = params.valid.paddingAmounts.borderAmounts[1].startEdgeSize
                pads[2] = params.valid.paddingAmounts.borderAmounts[0].endEdgeSize
                pads[3] = params.valid.paddingAmounts.borderAmounts[1].endEdgeSize
                # If padding amounts are all zero, there should be no padding list.
                if all(pad == 0 for pad in pads):
                    pads = None
                    auto_pad = 'VALID'
            else:
                auto_pad = 'VALID'

        elif pad_type == 'same':

            if params.same.asymmetryMode == SamePadding.BOTTOM_RIGHT_HEAVY:
                auto_pad = 'SAME_LOWER'
            elif params.same.asymmetryMode == SamePadding.TOP_LEFT_HEAVY:
                auto_pad = 'SAME_UPPER'
            else:
                raise ValueError('Unknown asymmetric mode: {}'.format(
                    params.same.asymmetryMode))

        else:
            raise ValueError('Unsupported padding mode: {}'.format(pad_type))

        if params.isDeconvolution and len(params.outputShape) > 0:
            nb.add_attribute('output_shape', params.outputShape)

        if pads is not None:
            nb.add_attribute('pads', pads)

        if auto_pad is not None:
            nb.add_attribute('auto_pad', auto_pad)

        strides = [1, 1]
        if len(params.stride) > 0:
            strides = params.stride
        nb.add_attribute('strides', strides)

        return nb.make_node()


registration.register_nn_converter('convolution', ConvolutionLayerConverter)
