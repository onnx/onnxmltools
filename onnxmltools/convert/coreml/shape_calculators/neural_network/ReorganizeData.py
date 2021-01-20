# SPDX-License-Identifier: Apache-2.0

import copy
from ....common._registration import register_shape_calculator
from ....common.data_types import FloatTensorType
from ....common.utils import check_input_and_output_numbers, check_input_and_output_types


def calculate_reorganize_data_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. [N, C, H, W] ---> [N, C * B * B , H / B, W / B]
        2. [N, C, H, W] ---> [N, C / B / B , H * B, W * B]

    Note taht B is the block size specified in this operator.
    '''
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])

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
