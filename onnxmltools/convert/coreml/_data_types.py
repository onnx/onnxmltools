from ...proto import onnx_proto

import numpy as np

class DataType(object):
    def __init__(self, type_name, shape=None):
        self.type_name = type_name
        self.shape = shape

    def __str__(self):
        return '%s' % self.type_name

    def to_onnx_type(self):
        raise NotImplementedError()


class Int64Type(DataType):
    def __init__(self):
        super(Int64Type, self).__init__('INT64', [1, 1])

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.INT64
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class FloatType(DataType):
    def __init__(self):
        super(FloatType, self).__init__('FLOAT', [1, 1])

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class StringType(DataType):
    def __init__(self):
        super(StringType, self).__init__('STRING', [1, 1])

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.STRING
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class TensorType(DataType):
    def __init__(self, type_name, element_type, shape=[]):
        super(TensorType, self).__init__(type_name, shape)
        self.element_type = element_type

    def __str__(self):
        return '%s-%s' % (str(self.shape), self.type_name)


class Int64TensorType(TensorType):
    def __init__(self, shape=[]):
        super(Int64TensorType, self).__init__('INT64S', Int64Type(), shape)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.INT64
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if isinstance(d, int):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = 'None'
            else:
                raise TypeError('Unsupported dimension value')
        return onnx_type


class FloatTensorType(TensorType):
    def __init__(self, shape=[], color_space=None):
        super(FloatTensorType, self).__init__('FLOATS', FloatType(), shape)
        self.color_space = color_space

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if isinstance(d, int):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = 'None'
            else:
                raise TypeError('Unsupported dimension value')
        return onnx_type


class StringTensorType(TensorType):
    def __init__(self, shape=[]):
        super(StringTensorType, self).__init__('STRINGS', StringType(), shape)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.STRING
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if isinstance(d, int):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = 'None'
            else:
                raise TypeError('Unsupported dimension value')
        return onnx_type


class DictionaryType(object):
    def __init__(self, key_type, value_type, max_cardinality=None):
        self.type_name = 'MAP'
        self.key_type = key_type
        self.value_type = value_type
        self.max_cardinality = max_cardinality  # It is only used to encode length of sparse vector. For other cases, it should be None.

    def __str__(self):
        return '%s-%s %s' % (str(self.key_type), str(self.value_type), self.type_name)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        if type(self.key_type) == Int64Type:
            onnx_type.map_type.key_type = onnx_proto.TensorProto.INT64
        elif type(self.key_type) == StringType:
            onnx_type.map_type.key_type = onnx_proto.TensorProto.STRING
        onnx_type.map_type.value_type.CopyFrom(self.value_type.to_onnx_type())
        return onnx_type


class SequenceType(DataType):
    def __init__(self, element_type, length='None'):
        super(SequenceType, self).__init__('SEQUENCE')
        self.element_type = element_type
        self.length = length

    def __str__(self):
        return '%s of %s' % (super(SequenceType, self).__str__(), str(self.element_type))

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.sequence_type.elem_type.CopyFrom(self.element_type.to_onnx_type())
        return onnx_type


def find_type_conversion(source_type, target_type):
    '''
    Find the operator name for converting source_type into target_type
    '''
    if type(source_type) == type(target_type):
        return 'identity'
    elif type(target_type) == FloatTensorType:
        return 'imageToFloatTensor'
    else:
        raise TypeError('Unsupported type conversion from %s to %s' % (source_type, target_type))


def parse_coreml_feature_type(raw_type):
    '''
    Extract type information from CoreML's FeatureType protobuf message
    '''
    type_name = raw_type.WhichOneof('Type')

    if type_name == 'int64Type':
        return Int64Type()
    elif type_name == 'doubleType':
        return FloatType()
    elif type_name == 'stringType':
        return StringType()
    elif type_name == 'imageType':
        # Produce [C, H, W]-tensor, where C is the number of color channels, H the height, and W the width.
        color_space = raw_type.imageType.colorSpace
        shape = ['None']
        if color_space == 10:  # gray scale
            shape.append(1)
        elif color_space in [20, 30]:  # RGB (20) or BGR (30)
            shape.append(3)
        else:
            raise RuntimeError('Invalid image color space')
        shape.append(raw_type.imageType.height)
        shape.append(raw_type.imageType.width)
        color_space_map = {10: 'GRAY', 20: 'RGB', 30: 'BGR'}
        return FloatTensorType(shape, color_space_map[color_space])
    elif type_name == 'multiArrayType':
        element_type_id = raw_type.multiArrayType.dataType
        shape = [d for d in raw_type.multiArrayType.shape]
        if len(shape) == 1:
            # [C]
            shape = ['None', shape[0]]
        elif len(shape) == 3:
            # [C, H, W]
            shape = ['None', shape[0], shape[1], shape[2]]
        else:
            shape = ['None', 1]  # Missing shape information. We will try inferring it.

        if element_type_id == 65568:
            # CoreML FLOAT32
            return FloatTensorType(shape)
        elif element_type_id == 65600:
            # CoreML DOUBLE
            return FloatTensorType(shape)
        elif element_type_id == 131104:
            # CoreML INT32
            return Int64TensorType(shape)
        else:
            raise RuntimeError('Invalid element type')
    elif type_name == 'dictionaryType':
        key_type = raw_type.dictionaryType.WhichOneof('KeyType')
        if key_type == 'int64KeyType':
            return DictionaryType(Int64Type(), FloatType())
        elif key_type == 'stringKeyType':
            return DictionaryType(StringType(), FloatType())
        else:
            raise RuntimeError('Unsupported key type: {}'.format(key_type))
    else:
        raise RuntimeError('Unsupported feature type: {}'.format(type_name))
