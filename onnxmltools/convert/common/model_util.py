#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from . import utils
from .NodeBuilder import NodeBuilder
from ...proto import onnx_proto
from ...proto import helper

# Includes basic ONNX data types only, advanced data types will be supported later.
# This list is used to convert integers to floats.
onnx_integer_types = [onnx_proto.TensorProto.UINT8, onnx_proto.TensorProto.INT8, onnx_proto.TensorProto.UINT16,
                      onnx_proto.TensorProto.INT16, onnx_proto.TensorProto.INT32, onnx_proto.TensorProto.INT64]

onnx_operator_version_info = {
    # Traditional machine learning operators
    'ArrayFeatureExtractor': ('ai.onnx.ml', 1),
    'Binarizer': ('ai.onnx.ml', 1),
    'CastMap': ('ai.onnx.ml', 1),
    'CategoryMapper': ('ai.onnx.ml', 1),
    'DictVectorizer': ('ai.onnx.ml', 1),
    'FeatureVectorizer': ('ai.onnx.ml', 1),
    'Imputer': ('ai.onnx.ml', 1),
    'LabelEncoder': ('ai.onnx.ml', 1),
    'LinearClassifier': ('ai.onnx.ml', 1),
    'LinearRegressor': ('ai.onnx.ml', 1),
    'Normalizer': ('ai.onnx.ml', 1),
    'OneHotEncoder': ('ai.onnx.ml', 1),
    'SVMClassifier': ('ai.onnx.ml', 1),
    'SVMRegressor': ('ai.onnx.ml', 1),
    'Scaler': ('ai.onnx.ml', 1),
    'TreeEnsembleClassifier': ('ai.onnx.ml', 1),
    'TreeEnsembleRegressor': ('ai.onnx.ml', 1),
    'ZipMap': ('ai.onnx.ml', 1),
    # Neural network operators
    'Abs': ('', 1),
    'Add': ('', 1),
    'And': ('', 1),
    'ArgMax': ('', 1),
    'ArgMin': ('', 1),
    'AveragePool': ('', 1),
    'BatchNormalization': ('', 1),
    'Cast': ('', 1),
    'Ceil': ('', 1),
    'Clip': ('', 1),
    'Concat': ('', 4),
    'Constant': ('', 1),
    'Conv': ('', 1),
    'ConvTranspose': ('', 1),
    'DepthToSpace': ('', 1),
    'Div': ('', 1),
    'Dropout': ('', 1),
    'Elu': ('', 1),
    'Equal': ('', 1),
    'Exp': ('', 1),
    'Flatten': ('', 1),
    'Floor': ('', 1),
    'GRU': ('', 3),
    'Gather': ('', 1),
    'Gemm': ('', 1),
    'GlobalAveragePool': ('', 1),
    'GlobalLpPool': ('', 2),
    'GlobalMaxPool': ('', 1),
    'Greater': ('', 1),
    'HardSigmoid': ('', 1),
    'Hardmax': ('', 1),
    'InstanceNormalization': ('', 1),
    'LRN': ('', 1),
    'LSTM': ('', 1),
    'LeakyRelu': ('', 1),
    'Less': ('', 1),
    'Log': ('', 1),
    'LogSoftmax': ('', 1),
    'LpNormalization': ('', 1),
    'LpPool': ('', 2),
    'MatMul': ('', 1),
    'Max': ('', 1),
    'MaxPool': ('', 1),
    'MaxRoiPool': ('', 1),
    'Mean': ('', 1),
    'Min': ('', 1),
    'Mul': ('', 1),
    'Neg': ('', 1),
    'Not': ('', 1),
    'Or': ('', 1),
    'PRelu': ('', 1),
    'Pad': ('', 2),
    'Pow': ('', 1),
    'RNN': ('', 1),
    'RandomNormal': ('', 1),
    'RandomNormalLike': ('', 1),
    'RandomUniform': ('', 1),
    'RandomUniformLike': ('', 1),
    'Reciprocal': ('', 1),
    'ReduceL1': ('', 1),
    'ReduceL2': ('', 1),
    'ReduceLogSum': ('', 1),
    'ReduceLogSumExp': ('', 1),
    'ReduceMax': ('', 1),
    'ReduceMean': ('', 1),
    'ReduceMin': ('', 1),
    'ReduceProd': ('', 1),
    'ReduceSum': ('', 1),
    'ReduceSumSquare': ('', 1),
    'Relu': ('', 1),
    'Reshape': ('', 1),
    'Selu': ('', 1),
    'Shape': ('', 1),
    'Sigmoid': ('', 1),
    'Size': ('', 1),
    'Slice': ('', 1),
    'Softmax': ('', 1),
    'Softplus': ('', 1),
    'Softsign': ('', 1),
    'SpaceToDepth': ('', 1),
    'Split': ('', 2),
    'Sqrt': ('', 1),
    'Squeeze': ('', 1),
    'Sub': ('', 1),
    'Sum': ('', 1),
    'Tanh': ('', 1),
    'Tile': ('', 1),
    'TopK': ('', 1),
    'Transpose': ('', 1),
    'Unsqueeze': ('', 1),
    'Xor': ('', 1),
    'ATen': ('', 1),
    'Affine': ('', 1),
    'ConstantFill': ('', 1),
    'Crop': ('', 1),
    'FC': ('', 1),
    'GRUUnit': ('', 1),
    'GivenTensorFill': ('', 1),
    'Identity': ('', 1),
    'If': ('', 1),
    'ImageScaler': ('', 1),
    'Loop': ('', 1),
    'LoopIndexTensor': ('', 1),
    'MeanVarianceNormalization': ('', 1),
    'ParametricSoftplus': ('', 1),
    'Scale': ('', 1),
    'ScaledTanh': ('', 1),
    'ThresholdedRelu': ('', 1),
    'Upsample': ('', 1)}


def make_tensor_value_info(name, elem_type=None, shape=None, doc_string=''):
    """
    Makes a TypeProto based on the data type and shape.
    """
    value_info_proto = onnx_proto.ValueInfoProto()
    value_info_proto.name = name

    if doc_string:
        value_info_proto.doc_string = doc_string

    if elem_type is not None or shape is not None:
        tensor_type_proto = value_info_proto.type.tensor_type
        if elem_type is not None:
            tensor_type_proto.elem_type = elem_type

        if shape is not None:
            tensor_shape_proto = tensor_type_proto.shape.dim
            for d in shape:
                dim = tensor_shape_proto.add()
                if utils.is_numeric_type(d):
                    dim.dim_value = d
                elif utils.is_string_type(d):
                    dim.dim_param = d
                else:
                    raise ValueError(
                        'Invalid item in shape: {}. '
                        'Needs to of integer_types or text_type.'.format(d))

    return value_info_proto


def make_sequence_value_info(name, elem_type, doc_string=''):
    """
    Makes a TypeProto based on the element type.
    """
    value_info_proto = onnx_proto.ValueInfoProto()
    value_info_proto.name = name

    if doc_string:
        value_info_proto.doc_string = doc_string

    sequence_type_proto = value_info_proto.type.sequence_type
    sequence_type_proto.elem_type.CopyFrom(elem_type)

    return value_info_proto


def make_map_value_info(name, key_type, value_type):
    """
    Makes a TypeProto based on the key/value types.
    """
    value_info_proto = onnx_proto.ValueInfoProto()
    value_info_proto.name = name

    map_type_proto = value_info_proto.type.map_type
    map_type_proto.key_type = key_type

    tensor_type_proto = map_type_proto.value_type.tensor_type
    tensor_type_proto.elem_type = value_type
    tensor_shape_proto = tensor_type_proto.shape.dim
    dim = tensor_shape_proto.add()
    dim.dim_value = 1

    return value_info_proto


def make_model(name, ir_version, producer, producer_version, domain, model_version, doc_string,
               nodes, inputs, outputs, values, initializer=list()):
    model = onnx_proto.ModelProto()
    model.ir_version = ir_version
    model.producer_name = producer
    model.producer_version = producer_version
    model.domain = domain
    model.model_version = model_version
    model.doc_string = doc_string
    for opset_domain, opset_version in set(onnx_operator_version_info[node.op_type] for node in nodes):
        opset = model.opset_import.add()
        opset.domain = opset_domain
        opset.version = opset_version
    graph = model.graph
    graph.name = name
    graph.node.extend(nodes)
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.value_info.extend(values)
    graph.initializer.extend(initializer)
    return model


def make_tensor(name, data_type, dims, vals, raw=False):
    return helper.make_tensor(name, data_type, dims, vals, raw)


def make_node(op_type, inputs, outputs, name=None, **kwargs):
    onnx_ml_ops = ["ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper", "DictVectorizer", "Imputer",
                   "FeatureVectorizer", "LabelEncoder", "LinearClassifier", "LinearRegressor", "Normalizer",
                   "OneHotEncoder", "Scaler", "SVMClassifier", "SVMRegressor", "TreeEnsembleClassifier",
                   "TreeEnsembleRegressor", "ZipMap"]

    node = helper.make_node(op_type, inputs, outputs, name, doc_string='', **kwargs)
    if op_type in onnx_ml_ops:
        node.domain = 'ai.onnx.ml'
    return node


def make_attribute(key, value, doc_string=None):
    helper.make_attribute(key, value, doc_string)


def make_zipmap_node(context, input, output, class_labels):
    '''
    Helper function to construct a ZipMap node
    '''
    from ..common import NodeBuilder
    nb = NodeBuilder(context, "ZipMap")
    if utils.is_string_type(class_labels):
        nb.add_attribute('classlabels_strings', class_labels)
    else:
        nb.add_attribute('classlabels_int64s', class_labels)

    nb.add_input(input)
    nb.add_output(output)
    return nb.make_node()


def make_normalizer_node(context, input, output, norm):
    '''
    Helper function to construct a normalizer node
    '''
    from ..common import NodeBuilder
    nb = NodeBuilder(context, "Normalizer")
    nb.add_attribute('norm', norm)
    nb.add_input(input)
    nb.add_output(output)
    return nb.make_node()


def get_tensorproto_typemap():
    '''
    get the typemap for all the tensor proto data types
    '''
    datatypes = [value.name for value in onnx_proto._TENSORPROTO_DATATYPE.values]
    typemap = dict((dt.lower(), getattr(onnx_proto.TensorProto, dt)) for dt in datatypes)
    return typemap


tensorproto_typemap = get_tensorproto_typemap()


def create_feature_extractor(input, output_name, indices, context):
    nb = NodeBuilder(context, 'ArrayFeatureExtractor')
    nb.add_input(input)

    tensor_dim = [1] if len(indices) == 1 else [1, len(indices)]
    index_tensor = make_tensor('TargetIndex', onnx_proto.TensorProto.INT64, tensor_dim, indices)
    nb.add_initializer(index_tensor)
    output = make_tensor_value_info(context.get_unique_name(output_name), shape=[1, len(indices)])
    nb.add_output(output)
    return nb.make_node()


def get_feature_count(input):
    if len(input.type.tensor_type.shape.dim) > 0:
        return input.type.tensor_type.shape.dim[-1].dim_value
    return 0


def create_feature_vector(inputs, output_name, context):
    input_names = []
    input_dims = []
    num_features = 0
    for inp in inputs:
        input_names.append(inp.name)
        feature_count = get_feature_count(inp)
        input_dims.append(feature_count)
        num_features += feature_count

    nb = NodeBuilder(context, 'FeatureVectorizer')
    nb.add_attribute('inputlist', input_names)
    nb.add_attribute('inputdimensions', input_dims)
    nb.extend_inputs(inputs)
    output = make_tensor_value_info(context.get_unique_name(output_name),
                                    onnx_proto.TensorProto.FLOAT, [1, num_features])
    nb.add_output(output)
    return nb.make_node()


def create_ohe(input, output_name, categories, context):
    nb = NodeBuilder(context, 'OneHotEncoder')
    nb.add_attribute('cats_int64s', categories)
    nb.add_input(input)
    output = make_tensor_value_info(context.get_unique_name(output_name),
                                    input.type.tensor_type.elem_type,
                                    shape=[1, len(categories)])
    nb.add_output(output)
    return nb.make_node()


def create_scaler(input, output_name, scale, offset, context):
    nb = NodeBuilder(context, "Scaler")
    nb.add_attribute('scale', [scale])
    nb.add_attribute('offset', [offset])

    nb.add_input(input)

    # Flatten out the input dims to create the tensor
    output_shape = [x.dim_value for x in input.type.tensor_type.shape.dim]
    output = make_tensor_value_info(context.get_unique_name(output_name), onnx_proto.TensorProto.FLOAT, output_shape)
    nb.add_output(output)
    return nb.make_node()
