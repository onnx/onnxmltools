# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numbers
from ...proto import onnx_proto


class DataType(object):
    def __init__(self, shape=None, doc_string=''):
        self.shape = shape
        self.doc_string = doc_string

    def to_onnx_type(self):
        raise NotImplementedError()


class Int64Type(DataType):
    def __init__(self, doc_string=''):
        super(Int64Type, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.INT64
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class FloatType(DataType):
    def __init__(self, doc_string=''):
        super(FloatType, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class StringType(DataType):
    def __init__(self, doc_string=''):
        super(StringType, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.STRING
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type


class TensorType(DataType):
    def __init__(self, shape=None, doc_string=''):
        super(TensorType, self).__init__([] if not shape else shape, doc_string)

    def _get_element_onnx_type(self):
        raise NotImplementedError()

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = self._get_element_onnx_type()
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if isinstance(d, numbers.Integral):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = 'None'
            else:
                raise ValueError('Unsupported dimension type: %s' % type(d))
        return onnx_type


class Int64TensorType(TensorType):
    def __init__(self, shape=None, doc_string=''):
        super(Int64TensorType, self).__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.INT64


class FloatTensorType(TensorType):
    def __init__(self, shape=None, color_space=None, doc_string=''):
        super(FloatTensorType, self).__init__(shape, doc_string)
        self.color_space = color_space

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.FLOAT


class StringTensorType(TensorType):
    def __init__(self, shape=None, doc_string=''):
        super(StringTensorType, self).__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.STRING


class DictionaryType(DataType):
    def __init__(self, key_type, value_type, shape=None, doc_string=''):
        super(DictionaryType, self).__init__(shape, doc_string)
        self.key_type = key_type
        self.value_type = value_type

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        if type(self.key_type) in [Int64Type, Int64TensorType]:
            onnx_type.map_type.key_type = onnx_proto.TensorProto.INT64
        elif type(self.key_type) in [StringType, StringTensorType]:
            onnx_type.map_type.key_type = onnx_proto.TensorProto.STRING
        onnx_type.map_type.value_type.CopyFrom(self.value_type.to_onnx_type())
        return onnx_type


class SequenceType(DataType):
    def __init__(self, element_type, shape=None, doc_string=''):
        super(SequenceType, self).__init__(shape, doc_string)
        self.element_type = element_type
        self.doc_string = doc_string

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
        raise ValueError('Unsupported type conversion from %s to %s' % (source_type, target_type))
