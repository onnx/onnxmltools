# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_upsample(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UpsampleLayerParams as Params
    op_type = 'Upsample'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    params = operator.raw_operator.upsample
    if params.mode == Params.NN:
        attrs['mode'] = 'NEAREST'
    elif params.mode == Params.BILINEAR:
        attrs['mode'] = 'BILINEAR'
    else:
        raise ValueError('Unsupported interpolation model in up-sampling')
    attrs['width_scale'] = float(params.scalingFactor[1])
    attrs['height_scale'] = float(params.scalingFactor[0])
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('upsample', convert_upsample)
