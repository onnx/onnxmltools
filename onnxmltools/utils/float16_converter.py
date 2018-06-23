#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
from struct import unpack
from ..proto import onnx_proto

def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_float_to_float16(model):
    '''
    Convert TensorProto DataType float in the ONNX ModelProto input to float16.

    :param model: ONNX ModelProto object
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)

        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')

    '''
    # raise error if input is empty or not ModelProto object
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))
    # create a queue for BFS
    queue = []
    queue.append(model)
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if hasattr(n, 'attribute'):
                    next_level.append(n.attribute)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            if isinstance(q, onnx_proto.AttributeProto):  # and hasattr(q, 'g'):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
                        # convert float_data (float) to float16 and write to int32_data
                        if n.float_data:
                            int_list = _npfloat16_to_int(np.float16(n.float_data))
                            n.int32_data[:] = int_list
                            n.float_data[:] = []
                        if n.raw_data:
                            # convert n.raw_data (bytes) to float
                            float32_list = []
                            for i in range(len(n.raw_data)//4):
                                float32_list += unpack('f', n.raw_data[i*4:(i+1)*4])
                            # convert float to float16
                            int_list = _npfloat16_to_int(np.float16(float32_list))
                            # convert float16 to bytes and write back to raw_data
                            n.raw_data = b''
                            for num in int_list:
                                n.raw_data += num.to_bytes(2, byteorder='little', signed=False)
                for n in q.input:
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                for n in q.output:
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                for n in q.value_info:
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # if q is node.attribute, process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                if q.t.data_type == onnx_proto.TensorProto.FLOAT:
                    q.t.data_type = onnx_proto.TensorProto.FLOAT16
                    if q.t.float_data:
                        int_list = _npfloat16_to_int(np.float16(q.t.float_data))
                        q.t.int32_data[:] = int_list
                        q.t.float_data[:] = []
                    if q.t.raw_data:
                        # convert q.t.raw_data (bytes) to float
                        float32_list = []
                        for i in range(len(q.t.raw_data) // 4):
                            float32_list += unpack('f', q.t.raw_data[i * 4:(i + 1) * 4])
                        # convert float to float16
                        int_list = _npfloat16_to_int(np.float16(float32_list))
                        # convert float16 to bytes and write back to raw_data
                        q.t.raw_data = b''
                        for num in int_list:
                            q.t.raw_data += num.to_bytes(2, byteorder='little', signed=False)
                for n in q.tensors:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
        queue = next_level

    return model

