import numpy as np
from ..convert.common.data_types import *
from onnxmltools.utils import load_model


def npfloat16_to_int(list):
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in list]


def float16_convert(file):
    model = load_model("C:\work\winmltools2\models\onnx_prerelease\winmlperf_coreml_SqueezeNet_prerelease.onnx")
    # model = load_model("C:\work\winmltools2\models\onnx_prerelease\winmlperf_coreml_MobileNet_prerelease.onnx")
    # model = load_model(file)
    # create a queue for BFS
    queue = []
    if not model.graph: return None
    queue.append(model)
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph
            try: q.graph
            except: pass
            else: next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute
            try: q.node
            except: pass
            else:
                for n in q.node:
                    try: n.attribute
                    except: pass
                    else: next_level.append(n.attribute)
            # if q is model.graph.node.attribute, push q.g and q.graphs
            try: q.g
            except: pass
            else: next_level.append(q.g)

            try: q.graphs
            except: pass
            else:
                for n in q.graphs:
                    next_level.append(n)
            # graph.initializer  (TensorProto)
            # for n in q.initializer:
            try: q.initializer
            except: pass
            else:
                for n in q.initializer:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
                        n.int32_data[:] = npfloat16_to_int(np.float16(n.float_data))
                        n.float_data[:] = []
                        # TODO: raw_data
            # graph.input (ValueInfoProto)
            try: q.input
            except: pass
            else:
                for n in q.input:
                    # check if the input has type of ValueInfoProto (GraphProto) or string (NodeProto)
                    try: n.type
                    except: pass
                    else:
                         if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # graph.output (ValueInfoProto)
            try: q.output
            except: pass
            else:
                for n in q.output:
                    # check if the output has type of ValueInfoProto (GraphProto) or string (NodeProto)
                    try: n.type
                    except: pass
                    else:
                        if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # if q is graph, process graph.value_info (ValueInfoProto)
            try: q.value_info
            except: pass
            else:
                for n in q.value_info:
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
            # if q is node.attribute, process node.attribute.t (TensorProto)
            try: q.t
            except: pass
            else:
                if q.t.data_type == onnx_proto.TensorProto.FLOAT:
                    q.t.data_type = onnx_proto.TensorProto.FLOAT16
                    q.t.int32_data[:] = npfloat16_to_int(np.float16(q.t.float_data))
                    q.t.float_data[:] = []
                    # TODO: raw_data
            # if q is node.attribute, process node.attribute.tensors (TensorProto)
            try: q.tensors
            except: pass
            else:
                for n in q.tensors:
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n.data_type = onnx_proto.TensorProto.FLOAT16
        queue = next_level
    return model
