#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from ...proto import onnx_proto
from ..common import model_util

_sklearn_type_map = {
    str: onnx_proto.TensorProto.STRING,
    int: onnx_proto.TensorProto.INT64,
    float: onnx_proto.TensorProto.FLOAT,
    np.dtype('float64'): onnx_proto.TensorProto.FLOAT,
}


def convert_type(sklearn_type):
    '''
    Converts an scikit-learn type to ONNX type
    '''
    if sklearn_type not in _sklearn_type_map:
        raise ValueError("scikit-learn type not supported: " + str(sklearn_type))
    return _sklearn_type_map[sklearn_type]


def convert_incoming_type(input_name, input_type, input_shape):
    if input_type in model_util.tensorproto_typemap:
        if not isinstance(input_shape, list):
            input_shape = [1, input_shape]
        elem_type = model_util.tensorproto_typemap[input_type]
        return model_util.make_tensor_value_info(input_name, elem_type, input_shape)

    if input_type.startswith('Dict'):
        split = input_type[4:].split('_')
        if len(split) != 2:
            raise ValueError("input_type variable should look like DictType1_Type2, current value is " + input_type)
        key_type = model_util.tensorproto_typemap[split[0]]
        onnx_value_type = model_util.tensorproto_typemap[split[1]]
        return model_util.make_map_value_info(input_name, key_type, onnx_value_type)

    if input_type.startswith('Multi'):
        data_type = model_util.tensorproto_typemap[input_type[4:]]
        onnx_shape = [1]
        for shape_val in input_shape:
            onnx_shape.append(shape_val)
        return model_util.make_tensor_value_info(input_name, data_type, onnx_shape)

    if input_type.startswith('Image'):
        data_type = onnx_proto.TensorProto.FLOAT
        return model_util.make_tensor_value_info(input_name, data_type, input_shape)
    raise ValueError("Don't know what to do with : " + str(input_type))
