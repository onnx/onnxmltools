# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
from ....common.data_types import FloatTensorType
from ....common._registration import register_shape_calculator


def calculate_reorganize_data_output_shapes(operator):
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise RuntimeError('Reorganize Data has only one input and one output')

    if type(operator.inputs[0].type) != FloatTensorType:
        raise RuntimeError('Input must be a float tensor')

    output_shape = copy.deepcopy(operator.inputs[0].type.shape)

    params = operator.raw_operator.reorganizeData

    from coremltools.proto.NeuralNetwork_pb2 import ReorganizeDataLayerParams as Params

    if params.mode == Params.DEPTH_TO_SPACE:
        if output_shape[1] % (params.blockSize * params.blockSize) != 0:
            raise RuntimeError('Channel number must be divisible by the square of block size')

        output_shape = [output_shape[0], output_shape[1] / params.blockSize / params.blockSize,
                        output_shape[2] * params.blockSize, output_shape[3] * params.blockSize]
    elif params.mode == Params.SPACE_TO_DEPTH:
        if output_shape[2] % params.blockSize != 0 or output_shape[3] % params.blockSize != 0:
            raise RuntimeError('Height and weight must be divisible by block size')

        output_shape = [output_shape[0], output_shape[1] * params.blockSize * params.blockSize,
                        output_shape[2] / params.blockSize, output_shape[3] / params.blockSize]
    else:
        raise ValueError('Unsupport reorganization mode {0}'.format(params.mode))

    operator.outputs[0].type = FloatTensorType([int(i) if i != 'None' else 'None' for i in output_shape],
                                               doc_string=operator.outputs[0].type.doc_string)


register_shape_calculator('reorganizeData', calculate_reorganize_data_output_shapes)
