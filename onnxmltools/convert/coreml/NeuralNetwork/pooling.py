#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ...common import NodeBuilder
from ...common import utils
from ...common import registration
from .reshape import extend_inputs_from_2d_to_4d
from coremltools.proto.NeuralNetwork_pb2 import PoolingLayerParams as Params
from coremltools.proto.NeuralNetwork_pb2 import SamePadding


class PoolingLayerConverter:
    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'pooling')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    # The conversion of pooling has several possible outcomes. Let's first define some symbols and then discuss their
    # ONNX computational graphs case-by-case.
    #
    # Symbols:
    #  X: input 4-D tensor. It should have a shape [N, C, H, W] following CoreML's pooling definition.
    #  Y: output tensor identical to CorML's pooling. Its shapes depends on the pooling type applied.
    #
    # Case 1: global pooling
    #  X ---> ONNX Global Pooling ---> Y
    # In this case, ONNX's pooling implementation should directly match CoreML's implementation, so it's just a naive
    # translation.
    #
    # Case 2: local max/L2 pooling with same or valid padding
    #
    #  X ---> ONNX Local Max/L2 Pooling ---> Y
    #
    # In this case, ONNX's pooling implementation should directly match CoreML's implementation, so it's just a naive
    # translation.
    #
    # Case 3: local max/L2 pooling under CoreML's IncludeLastPixel padding
    #
    # This case is not supported in ONNX; there is no IncludeLastPixel padding in ONNX
    #
    # Case 4: local average pooling with same or valid padding. exclude_pad_area is on.
    #
    #  X ---> ONNX Local Average Pooling ---> Y
    #
    # Current ONNX pooling operator just follows Caffe2, so the padded area is naturally excluded when calculating the
    # numerator and denumerator of the pixel average covered by the kernel. That is, the translation from CoreML to ONNX
    # is trivial.
    #
    # Case 5: local average pooling with same or valid padding. exclude_pad_area is off.
    #
    #  X ---> ONNX Local Average Pooling ---> Y' ------------> Mul ---> Y
    #  |                                                        ^
    #  |                                                        |
    #  '---> Scaler ---> Z ---> ONNX L1-norm Pooling ---> Z' ---'
    #
    #  The Scaler has "alpha=0" and its "beta" is a constant. If "beta=1", the output of the L1-norm pooling, Z', is the
    #  effective kernel size applied at each pixel when padded area is excluded. Here we use "beta=1/kerenel_size" so
    #  that one value in Z' stands for
    #         (the kernel size without padded area) / (the kernel size with padded area)
    #  at a pixel. The output Y' is computed with exclude_pad_area=on, so the element-wise multiplication of Y' and Z'
    #  is Y.
    #
    # Case 6: local average pooling with IncludeLastPixel padding. exclude_pad_area is on.
    #
    # This case is not supported due to the use of IncludeLastPixel padding.
    #
    # Case 7: local average pooling with IncludeLastPixel padding. exclude_pad_area is off.
    #
    # This case is not supported due to the use of IncludeLastPixel padding.
    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        extend_inputs_from_2d_to_4d(context, inputs)

        params = cm_node.pooling

        # Handle global pooling mode
        if params.globalPooling:
            pooling_table = {params.MAX: 'GlobalMaxPool',
                             Params.AVERAGE: 'GlobalAveragePool',
                             Params.L2: 'GlobalLpPool'}

            if params.type not in pooling_table:
                raise ValueError('Unsupported pooling type: {}'.format(params.type))

            nb = NodeBuilder(context, pooling_table[params.type])
            if params.type == Params.L2:
                nb.add_attribute('p', 2)

            nb.extend_inputs(inputs)
            nb.extend_outputs(outputs)

            return nb.make_node()

        # Handle local pooling mode
        kernel_shape = [3, 3]
        if len(params.kernelSize) > 0:
            kernel_shape = params.kernelSize
        strides = [1, 1]
        if len(params.stride) > 0:
            strides = params.stride
        if params.type == Params.MAX:
            nb = NodeBuilder(context, 'MaxPool')
            nb.add_attribute('dilations', [1, 1])
            nb.add_attribute('kernel_shape', kernel_shape)
            nb.add_attribute('strides', strides)
        elif params.type == Params.AVERAGE:
            nb = NodeBuilder(context, 'AveragePool')
            nb.add_attribute('kernel_shape', kernel_shape)
            nb.add_attribute('strides', strides)
        elif params.type == Params.L2:
            nb = NodeBuilder(context, 'LpPool')
            nb.add_attribute('kernel_shape', kernel_shape)
            nb.add_attribute('strides', strides)
            nb.add_attribute('p', 2)
        else:
            raise ValueError('Unsupported pooling type: {}'.format(params.type))

        # Set up padding attributes
        pads = None
        auto_pad = None
        pad_type = params.WhichOneof('PoolingPaddingType')
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
                raise ValueError('Unknown asymmetric mode: {}'.format(params.same.asymmetryMode))

        else:
            raise ValueError('Unsupported padding mode: {}'.format(pad_type))

        if pads is not None:
            nb.add_attribute('pads', pads)

        if auto_pad is not None:
            nb.add_attribute('auto_pad', auto_pad)

        nb.extend_inputs(inputs)

        builders = [nb]
        if params.type == Params.AVERAGE and not params.avgPoolExcludePadding:
            # Case 5. See comment above.
            pooled_buffer_name = context.get_unique_name('pooled_buffer')
            nb.add_output(pooled_buffer_name)

            constant_tensor_name = context.get_unique_name('constant')
            kernel_size_map = context.get_unique_name('kernel_size_map')
            scaler_builder = NodeBuilder(context, 'Affine')
            scaler_builder.add_attribute('alpha', 0.)
            scaler_builder.add_attribute('beta', 1. / (kernel_shape[0] * kernel_shape[1]))
            scaler_builder.add_input(inputs[0])
            scaler_builder.add_output(constant_tensor_name)
            builders.append(scaler_builder)

            lp_pool_builder = NodeBuilder(context, 'LpPool')
            lp_pool_builder.add_attribute('kernel_shape', kernel_shape)
            lp_pool_builder.add_attribute('strides', strides)
            lp_pool_builder.add_attribute('p', 1)
            if pads is not None:
                lp_pool_builder.add_attribute('pads', pads)
            if auto_pad is not None:
                lp_pool_builder.add_attribute('auto_pad', auto_pad)
            lp_pool_builder.add_input(constant_tensor_name)
            lp_pool_builder.add_output(kernel_size_map)
            builders.append(lp_pool_builder)

            adjuster_builder = NodeBuilder(context, 'Mul')
            adjuster_builder.add_input(pooled_buffer_name)
            adjuster_builder.add_input(kernel_size_map)
            adjuster_builder.add_output(outputs[0])
            builders.append(adjuster_builder)
        else:
            nb.extend_outputs(outputs)

        return [builder.make_node() for builder in builders if builder is not None]


registration.register_nn_converter('pooling', PoolingLayerConverter)
