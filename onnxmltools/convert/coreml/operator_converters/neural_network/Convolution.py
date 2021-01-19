# SPDX-License-Identifier: Apache-2.0

from .....proto import onnx_proto
from ....common._registration import register_converter


def convert_convolution(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import SamePadding

    params = operator.raw_operator.convolution
    op_type = 'ConvTranspose' if params.isDeconvolution else 'Conv'
    inputs = [operator.inputs[0].full_name]
    outputs = [operator.outputs[0].full_name]
    attrs = {'name': operator.full_name}

    n_groups = 1 if params.nGroups == 0 else params.nGroups

    shape_w = [params.outputChannels, params.kernelChannels, params.kernelSize[0], params.kernelSize[1]]
    if params.isDeconvolution:
        shape_w[0] = params.kernelChannels
        shape_w[1] = int(params.outputChannels / n_groups)
    name_w = scope.get_unique_variable_name(operator.full_name + '_W')
    inputs.append(name_w)
    container.add_initializer(name_w, onnx_proto.TensorProto.FLOAT, shape_w, params.weights.floatValue)

    if params.hasBias:
        shape_b = [len(params.bias.floatValue)]
        name_b = scope.get_unique_variable_name(operator.full_name + '_B')
        inputs.append(name_b)
        container.add_initializer(name_b, onnx_proto.TensorProto.FLOAT, shape_b, params.bias.floatValue)

    dilations = [1, 1]
    if len(params.dilationFactor) > 0:
        dilations = [params.dilationFactor[0], params.dilationFactor[1]]
    kernel_shape = [3, 3]
    if len(params.kernelSize) > 0:
        kernel_shape = params.kernelSize
    strides = [1, 1]
    if len(params.stride) > 0:
        strides = params.stride
    attrs['dilations'] = dilations
    attrs['group'] = n_groups
    attrs['kernel_shape'] = kernel_shape
    attrs['strides'] = strides

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
            auto_pad = 'SAME_UPPER'
        elif params.same.asymmetryMode == SamePadding.TOP_LEFT_HEAVY:
            auto_pad = 'SAME_LOWER'
        else:
            raise ValueError('Unknown asymmetric mode: {}'.format(
                params.same.asymmetryMode))
    else:
        raise ValueError('Unsupported padding mode: {}'.format(pad_type))

    if params.isDeconvolution and len(params.outputShape) > 0:
        attrs['output_shape'] = params.outputShape

    if pads is not None:
        attrs['pads'] = pads

    if auto_pad is not None:
        attrs['auto_pad'] = auto_pad

    container.add_node(op_type, inputs, outputs, **attrs)


register_converter('convolution', convert_convolution)
