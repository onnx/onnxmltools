import numpy as np
from struct import unpack
from ..convert.common.data_types import *
from onnxmltools.utils import load_model, save_model


def npfloat16_to_int(np_list):
    """
    Convert numpy float16 to python int.
    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def float16_converter(onnx_file, new_onnx_file):
    """
    Convert ONNX model from TensorProto type float to float16.
    :param onnx_file: ONNX model path and name
    :param new_onnx_file: new ONNX model path and name

    Example:

    ::

        from onnxmltools.utils import float16_converter
        onnx_file = 'c:/winmlperf_coreml_SqueezeNet_prerelease.onnx'
        new_onnx_file = 'c:/winmlperf_coreml_SqueezeNet_prerelease_float16.onnx'
        float16_converter(onnx_file, new_onnx_file)
    """
    # read onnx file to model
    model = load_model(onnx_file)
    # create a queue for BFS
    queue = []
    if not model.graph: return None
    queue.append(model)
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph
            if hasattr(q, 'graph'):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute
            if hasattr(q, 'node'):
                for n in q.node:
                    if hasattr(n, 'attribute'):
                        next_level.append(n.attribute)
            # if q is model.graph.node.attribute, push q.g and q.graphs
            if hasattr(q, 'g'):
                next_level.append(q.g)

            if hasattr(q, 'graphs'):
                for n in q.graphs:
                    next_level.append(n)
            # if q is graph, process graph.initializer(TensorProto), input and output (ValueInfoProto)
            if hasattr(q, 'initializer'):
                for n in q.initializer:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
                        # convert float_data (float) to float16 and write to int32_data
                        int_list = npfloat16_to_int(np.float16(n.float_data))
                        n.int32_data[:] = int_list
                        n.float_data[:] = []
                        # convert n.raw_data (bytes) to float
                        float32_list = []
                        for i in range(len(n.raw_data)//4):
                            float32_list += unpack('f', n.raw_data[i*4:(i+1)*4])
                        # convert float to float16
                        int_list = npfloat16_to_int(np.float16(float32_list))
                        # convert float16 to bytes and write back to raw_data
                        n.raw_data = b''
                        for num in int_list:
                            n.raw_data += num.to_bytes(2, byteorder='little', signed=True)
            # graph.input (ValueInfoProto)
            if hasattr(q, 'input'):
                for n in q.input:
                    # check if the input has type of ValueInfoProto (GraphProto) or string (NodeProto)
                    if hasattr(n, 'type'):
                         if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # graph.output (ValueInfoProto)
            if hasattr(q, 'output'):
                for n in q.output:
                    # check if the output has type of ValueInfoProto (GraphProto) or string (NodeProto)
                    if hasattr(n, 'type'):
                        if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # if q is graph, process graph.value_info (ValueInfoProto)
            if hasattr(q, 'value_info'):
                for n in q.value_info:
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # if q is node.attribute, process node.attribute.t (TensorProto)
            if hasattr(q, 't'):
                if q.t.data_type == onnx_proto.TensorProto.FLOAT:
                    q.t.data_type = onnx_proto.TensorProto.FLOAT16
                    int_list = npfloat16_to_int(np.float16(q.t.float_data))
                    q.t.int32_data[:] = int_list
                    q.t.float_data[:] = []
                    # convert q.t.raw_data (bytes) to float
                    float32_list = []
                    for i in range(len(q.t.raw_data) // 4):
                        float32_list += unpack('f', q.t.raw_data[i * 4:(i + 1) * 4])
                    # convert float to float16
                    int_list = npfloat16_to_int(np.float16(float32_list))
                    # convert float16 to bytes and write back to raw_data
                    q.t.raw_data = b''
                    for num in int_list:
                        q.t.raw_data += num.to_bytes(2, byteorder='little', signed=True)
            # if q is node.attribute, process node.attribute.tensors (TensorProto)
            if hasattr(q, 'tensors'):
                for n in q.tensors:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
        queue = next_level
    # save converted model to new onnx file
    save_model(model, new_onnx_file)

