from ...proto import onnx_proto


class DataType(object):
    def __init__(self, type_name, shape=None, doc_string=''):
        self.type_name = type_name
        self.shape = shape
        self.doc_string = doc_string

    def __str__(self):
        return '%s' % self.type_name

    def to_onnx_type(self):
        raise NotImplementedError()


class Int64Type(DataType):
    def __init__(self, doc_string=''):
        super(Int64Type, self).__init__('INT64', [1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.INT64
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class FloatType(DataType):
    def __init__(self, doc_string=''):
        super(FloatType, self).__init__('FLOAT', [1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class StringType(DataType):
    def __init__(self, doc_string=''):
        super(StringType, self).__init__('STRING', [1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.STRING
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class TensorType(DataType):
    def __init__(self, type_name, element_type, shape=[], doc_string=''):
        super(TensorType, self).__init__(type_name, shape, doc_string)
        self.element_type = element_type

    def __str__(self):
        return '%s-%s' % (str(self.shape), self.type_name)


class Int64TensorType(TensorType):
    def __init__(self, shape=[], doc_string=''):
        super(Int64TensorType, self).__init__('INT64S', Int64Type(), shape, doc_string)

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
    def __init__(self, shape=[], color_space=None, doc_string=''):
        super(FloatTensorType, self).__init__('FLOATS', FloatType(), shape, doc_string)
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
    def __init__(self, shape=[], doc_string=''):
        super(StringTensorType, self).__init__('STRINGS', StringType(), shape, doc_string)

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
    # [TODO] Use DataType as base of all types
    def __init__(self, key_type, value_type, max_cardinality=None, doc_string=''):
        self.type_name = 'MAP'
        self.key_type = key_type
        self.value_type = value_type
        self.max_cardinality = max_cardinality  # It is only used to encode length of sparse vector. For other cases, it should be None.
        self.doc_string = doc_string

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
    def __init__(self, element_type, length='None', doc_string=''):
        super(SequenceType, self).__init__('SEQUENCE')
        self.element_type = element_type
        self.length = length
        self.doc_string = doc_string

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


def parse_coreml_feature(feature_info, batch_size=1):
    '''
    Encode type information from CoreML's FeatureType protobuf message in converter's type system.

    Scalar types such as Int64FeatureType, DoubleFeatureType, and StringFeatureType in CoreML are interpreted as
    [batch_size, 1]-tensor. Tensor-like types such as ArrayFeature in CoreML is viewed as tensors with a prepend
    batch_size; for example, we use [batch_size, C, H, W] to denote [C, H, W]-array in CoreML.
    :param feature_info: CoreML FeatureDescription (https://apple.github.io/coremltools/coremlspecification/sections/DataStructuresAndFeatureTypes.html#featuretype)
    :param batch_size: default batch size prepend to scalars and tensors variables from CoreML
    :return: one of our Int64Type, FloatType, StringType, Int64TensorType, FloatTensorType, or DictionaryType
    '''
    raw_type = feature_info.type
    doc_string = feature_info.shortDescription
    type_name = raw_type.WhichOneof('Type')

    if type_name == 'int64Type':
        return Int64Type(doc_string=doc_string)
    elif type_name == 'doubleType':
        return FloatType(doc_string=doc_string)
    elif type_name == 'stringType':
        return StringType(doc_string=doc_string)
    elif type_name == 'imageType':
        # Produce [C, H, W]-tensor, where C is the number of color channels, H the height, and W the width.
        color_space = raw_type.imageType.colorSpace
        shape = [batch_size]
        if color_space == 10:  # gray scale
            shape.append(1)
            doc_string = 'Image(s) in gray scale. If there are N images, it is a 4-D tensor with shape [N, 1, H, W]'
        elif color_space == 20:  # RGB (20)
            shape.append(3)
            doc_string = 'Image(s) in RGB format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                         'C-axis are red, green, and blue channels, respectively.'
        elif color_space == 30:  # BGR (30)
            shape.append(3)
            doc_string = 'Image(s) in BGR format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                         'C-axis are blue, green, and red channels, respectively.'
        else:
            raise RuntimeError('Unknown image format. Only gray-level, RGB, and BGR are supported')
        shape.append(raw_type.imageType.height)
        shape.append(raw_type.imageType.width)
        color_space_map = {10: 'GRAY', 20: 'RGB', 30: 'BGR'}
        return FloatTensorType(shape, color_space_map[color_space], doc_string=doc_string)
    elif type_name == 'multiArrayType':
        element_type_id = raw_type.multiArrayType.dataType
        shape = [d for d in raw_type.multiArrayType.shape]
        if len(shape) == 1:
            # [C]
            shape = [batch_size, shape[0]]
        elif len(shape) == 3:
            # [C, H, W]
            shape = [batch_size, shape[0], shape[1], shape[2]]
        else:
            shape = [batch_size, 1]  # Missing shape information. We will try inferring it.

        if element_type_id in [65568, 65600]:
            # CoreML FLOAT32 & DOUBLE
            return FloatTensorType(shape, doc_string=doc_string)
        elif element_type_id == 131104:
            # CoreML INT32
            return Int64TensorType(shape, doc_string=doc_string)
        else:
            raise RuntimeError('Invalid element type')
    elif type_name == 'dictionaryType':
        key_type = raw_type.dictionaryType.WhichOneof('KeyType')
        if key_type == 'int64KeyType':
            return DictionaryType(Int64Type(), FloatType(), doc_string=doc_string)
        elif key_type == 'stringKeyType':
            return DictionaryType(StringType(), FloatType(), doc_string=doc_string)
        else:
            raise RuntimeError('Unsupported key type: {}'.format(key_type))
    else:
        raise RuntimeError('Unsupported feature type: {}'.format(type_name))
