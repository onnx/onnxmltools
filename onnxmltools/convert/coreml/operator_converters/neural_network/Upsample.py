# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxtk.common._apply_operation import apply_upsample
from onnxtk.common._registration import register_converter


def convert_upsample(scope, operator, container):
    from coremltools.proto.NeuralNetwork_pb2 import UpsampleLayerParams as Params
    params = operator.raw_operator.upsample

    if params.mode == Params.NN:
        mode = 'NEAREST'
    elif params.mode == Params.BILINEAR:
        mode = 'BILINEAR'
    else:
        raise ValueError('Unsupported interpolation model in up-sampling')

    width_scale = float(params.scalingFactor[1])
    height_scale = float(params.scalingFactor[0])

    apply_upsample(scope, operator.input_full_names, operator.output_full_names, container, operator_name=None,
                   mode=mode, scales=[1, 1, height_scale, width_scale])


register_converter('upsample', convert_upsample)
