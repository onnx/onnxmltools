#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import numpy as np
import itertools
from struct import unpack
from ..proto import onnx_proto
from ..convert.common._container import ModelComponentContainer
import onnx
from onnx.shape_inference import infer_shapes



def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_float_to_float16(model):
    '''
    Convert tensor float in the ONNX ModelProto or TensorProto to tensor float16.

    :param model: ONNX ModelProto object or TensorProto object
    :return: converted ONNX ModelProto or TensorProto object

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
    domain_flag = 0
    if isinstance(model, onnx_proto.ModelProto):
        # create black list
        op_black_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                         'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor', 'Normalizer',
                         'OneHotEncoder', 'SVMClassifier', 'SVMRegressor', 'Scaler', 'TreeEnsembleClassifier',
                         'TreeEnsembleRegressor', 'ZipMap']
        # create a queue for BFS
        queue = []
        value_info_list = []
        node_list = []
        # type inference on input model
        model = infer_shapes(model)
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
                        # if n is in the black list (doesn't support float16), no conversion for the node,
                        # and save the node for further processing
                        if n.op_type in op_black_list:
                            node_list.append(n)
                        else:
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
                    # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                    # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                    for n in itertools.chain(q.input, q.output, q.value_info):
                        if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                            value_info_list.append(n)
                # if q is node.attribute, process node.attribute.t and node.attribute.tensors (TensorProto)
                if isinstance(q, onnx_proto.AttributeProto):
                    for n in itertools.chain(q.t, q.tensors):
                        if n.data_type == onnx_proto.TensorProto.FLOAT:
                            n.data_type = onnx_proto.TensorProto.FLOAT16
                            if n.float_data:
                                int_list = _npfloat16_to_int(np.float16(n.float_data))
                                n.int32_data[:] = int_list
                                n.float_data[:] = []
                            if n.raw_data:
                                # convert n.raw_data (bytes) to float
                                float32_list = []
                                for i in range(len(n.raw_data) // 4):
                                    float32_list += unpack('f', n.raw_data[i * 4:(i + 1) * 4])
                                # convert float to float16
                                int_list = _npfloat16_to_int(np.float16(float32_list))
                                # convert float16 to bytes and write back to raw_data
                                n.raw_data = b''
                                for num in int_list:
                                    n.raw_data += num.to_bytes(2, byteorder='little', signed=False)
            queue = next_level

        # process the nodes in black list that doesn't support tensor(float16)
        for node in node_list:
            # if input's name is in the value_info_list meaning input is tensor(float16) type, insert a Cast node
            # before the node, change current node's input name and create new value_info for the new name
            for i in range(len(node.input)):
                input = node.input[i]
                for value_info in value_info_list:
                    if input == value_info.name:
                        # create new value_info for current node's new input name
                        new_value_info = onnx_proto.ValueInfoProto()
                        new_value_info.CopyFrom(value_info)
                        new_value_info.name = input + '_casted'
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                        model.graph.value_info._values.append(new_value_info)
                        # add Cast node (from tensor(float16) to tensor(float) before current node
                        container = ModelComponentContainer(onnx.__version__)
                        attrs = {'name': input + 'Cast'}
                        attrs['to'] = onnx_proto.TensorProto.FLOAT
                        container.add_node('Cast', input, input + '_casted', op_domain='', op_version=7, **attrs)
                        model.graph.node.extend(container.nodes)
                        # change current node's input name
                        node.input[i] = input + '_casted'
                        domain_flag = 1
                        continue
            # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float16 to
            # float Cast node after the node, change current node's output name and create new value_info for the new name
            for i in range(len(node.output)):
                output = node.output[i]
                for value_info in value_info_list:
                    if output == value_info.name:
                        # create new value_info for current node's new output
                        new_value_info = onnx_proto.ValueInfoProto()
                        new_value_info.CopyFrom(value_info)
                        new_value_info.name = output + '_casted'
                        new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                        model.graph.value_info._values.append(new_value_info)
                        # add Cast node (from tensor(float) to tensor(float16) after current node
                        container = ModelComponentContainer(onnx.__version__)
                        attrs = {'name': output + 'Cast'}
                        attrs['to'] = onnx_proto.TensorProto.FLOAT16
                        container.add_node('Cast', output + '_casted', output, op_domain='', op_version=7, **attrs)
                        model.graph.node.extend(container.nodes)
                        # change current node's input name
                        node.output[i] = output + '_casted'
                        domain_flag = 1
                        continue
        if domain_flag:
            # Create operator set for cast node
            op_set = model.opset_import.add()
            op_set.domain = ""
            op_set.version = 7
        return model
    elif isinstance(model, onnx_proto.TensorProto):
        # create a queue for BFS
        queue = []
        queue.append(model)
        while queue:
            next_level = []
            for q in queue:
                # if q is TensorProto
                if q.data_type == onnx_proto.TensorProto.FLOAT:
                    q.data_type = onnx_proto.TensorProto.FLOAT16
                    # convert float_data (float) to float16 and write to int32_data
                    if q.float_data:
                        int_list = _npfloat16_to_int(np.float16(q.float_data))
                        q.int32_data[:] = int_list
                        q.float_data[:] = []
                    # convert raw_data (bytes) to float
                    if q.raw_data:
                        float32_list = []
                        for i in range(len(q.raw_data) // 4):
                            float32_list += unpack('f', q.raw_data[i * 4:(i + 1) * 4])
                        # convert float to float16
                        int_list = _npfloat16_to_int(np.float16(float32_list))
                        # convert float16 to bytes and write back to raw_data
                        q.raw_data = b''
                        for num in int_list:
                            q.raw_data += num.to_bytes(2, byteorder='little', signed=False)
            queue = next_level
        return model
    else:
        raise ValueError('Expected model type is an ONNX ModelProto or TensorProto but got %s' % type(model))
