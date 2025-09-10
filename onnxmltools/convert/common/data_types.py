# SPDX-License-Identifier: Apache-2.0

import numbers
import numpy as np
import onnx


class DataType:
    def __init__(self, shape=None, doc_string=""):
        self.shape = shape
        self.doc_string = doc_string

    def to_onnx_type(self):
        raise NotImplementedError()

    def __repr__(self):
        return "{}(shape={})".format(self.__class__.__name__, self.shape)


class FloatType(DataType):
    def __init__(self, doc_string=""):
        super(FloatType, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Int64Type(DataType):
    def __init__(self, doc_string=""):
        super(Int64Type, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.INT64
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class StringType(DataType):
    def __init__(self, doc_string=""):
        super(StringType, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.STRING
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class TensorType(DataType):
    def __init__(
        self, shape=None, doc_string="", denotation=None, channel_denotations=None
    ):
        super(TensorType, self).__init__(shape if shape else [], doc_string)
        self.denotation = denotation
        self.channel_denotations = channel_denotations

    def _get_element_onnx_type(self):
        raise NotImplementedError()

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = self._get_element_onnx_type()
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if d is None:
                pass
            elif isinstance(d, numbers.Integral):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = d
            else:
                raise ValueError(
                    "Unsupported dimension type: %s, see %s"
                    % (
                        type(d),
                        "https://github.com/onnx/onnx/blob/master/docs/IR.md#"
                        + "input--output-data-types",
                    )
                )
        if getattr(onnx_type, "denotation", None) is not None:
            if self.denotation:
                onnx_type.denotation = self.denotation
            if self.channel_denotations:
                for d, denotation in zip(
                    onnx_type.tensor_type.shape.dim, self.channel_denotations
                ):
                    if denotation:
                        d.denotation = denotation
        return onnx_type


class DoubleType(DataType):
    def __init__(self, doc_string=""):
        super().__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class SequenceType(DataType):
    def __init__(self, element_type, shape=None, doc_string=""):
        super(SequenceType, self).__init__(shape, doc_string)
        self.element_type = element_type
        self.doc_string = doc_string

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        try:
            onnx_type.sequence_type.elem_type.CopyFrom(self.element_type.to_onnx_type())
        except AttributeError as ee:
            msg = "ONNX was not compiled with flag ONNX-ML.\n{0}\n{1}"
            msg = msg.format(str(self), str(self.element_type.to_onnx_type()))
            info = [onnx.__version__, str(onnx_type)]
            msg += "\n".join(info)
            raise RuntimeError(msg) from ee
        except TypeError as e:
            raise RuntimeError(
                "Unable to create SequenceType with "
                "element_type=%r" % self.element_type
            ) from e
        return onnx_type

    def __repr__(self):
        return "SequenceType(element_type={0})".format(self.element_type)


class DictionaryType(DataType):
    def __init__(self, key_type, value_type, shape=None, doc_string=""):
        super(DictionaryType, self).__init__(shape, doc_string)
        self.key_type = key_type
        self.value_type = value_type

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        try:
            if type(self.key_type) in [Int64Type, Int64TensorType]:
                onnx_type.map_type.key_type = onnx.TensorProto.INT64
            elif type(self.key_type) in [StringType, StringTensorType]:
                onnx_type.map_type.key_type = onnx.TensorProto.STRING
            onnx_type.map_type.value_type.CopyFrom(self.value_type.to_onnx_type())
        except AttributeError as e:
            msg = "ONNX was not compiled with flag ONNX-ML.\n{0}\n{1}"
            msg = msg.format(str(self), str(self.value_type.to_onnx_type()))
            info = [onnx.__version__, str(onnx_type)]
            msg += "\n".join(info)
            raise RuntimeError(msg) from e
        return onnx_type

    def __repr__(self):
        return "DictionaryType(key_type={0}, value_type={1})".format(
            self.key_type, self.value_type
        )


class BooleanTensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.BOOL


class Complex64TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.COMPLEX64


class Complex128TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.COMPLEX128


class FloatTensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.FLOAT


class StringTensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.STRING


class DoubleTensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.DOUBLE


class Float16TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.FLOAT16


class Int8TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.INT8


class Int16TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.INT16


class Int32TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.INT32


class Int64TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.INT64


class UInt16TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.UINT16


class UInt32TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.UINT32


class UInt64TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.UINT64


class UInt8TensorType(TensorType):
    def _get_element_onnx_type(self):
        return onnx.TensorProto.UINT8


class UInt8Type(DataType):
    def __init__(self, doc_string=""):
        super(UInt8Type, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.UINT8
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class Int8Type(DataType):
    def __init__(self, doc_string=""):
        super(Int8Type, self).__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx.TypeProto()
        onnx_type.tensor_type.elem_type = onnx.TensorProto.INT8
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def copy_type(vtype, empty=True):
    if isinstance(vtype, SequenceType):
        return vtype.__class__(copy_type(vtype.element_type))
    if isinstance(vtype, DictionaryType):
        return vtype.__class__(copy_type(vtype.key_type), copy_type(vtype.value_type))
    return vtype.__class__()


def _guess_type_proto(data_type, dims):
    for d in dims:
        if d == 0:
            raise RuntimeError("Dimension should not be null: {}.".format(list(dims)))
    if data_type == onnx.TensorProto.FLOAT:
        return FloatTensorType(dims)
    if data_type == onnx.TensorProto.DOUBLE:
        return DoubleTensorType(dims)
    if data_type == onnx.TensorProto.STRING:
        return StringTensorType(dims)
    if data_type == onnx.TensorProto.INT64:
        return Int64TensorType(dims)
    if data_type == onnx.TensorProto.INT32:
        return Int32TensorType(dims)
    if data_type == onnx.TensorProto.BOOL:
        return BooleanTensorType(dims)
    if data_type == onnx.TensorProto.INT8:
        return Int8TensorType(dims)
    if data_type == onnx.TensorProto.UINT8:
        return UInt8TensorType(dims)
    if Complex64TensorType is not None:
        if data_type == onnx.TensorProto.COMPLEX64:
            return Complex64TensorType(dims)
        if data_type == onnx.TensorProto.COMPLEX128:
            return Complex128TensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type)
    )


def _guess_type_proto_str(data_type, dims):
    if data_type == "tensor(float)":
        return FloatTensorType(dims)
    if data_type == "tensor(double)":
        return DoubleTensorType(dims)
    if data_type == "tensor(string)":
        return StringTensorType(dims)
    if data_type == "tensor(int64)":
        return Int64TensorType(dims)
    if data_type == "tensor(int32)":
        return Int32TensorType(dims)
    if data_type == "tensor(bool)":
        return BooleanTensorType(dims)
    if data_type == "tensor(int8)":
        return Int8TensorType(dims)
    if data_type == "tensor(uint8)":
        return UInt8TensorType(dims)
    if Complex64TensorType is not None:
        if data_type == "tensor(complex64)":
            return Complex64TensorType(dims)
        if data_type == "tensor(complex128)":
            return Complex128TensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type)
    )


def _guess_type_proto_str_inv(data_type):
    if isinstance(data_type, FloatTensorType):
        return "tensor(float)"
    if isinstance(data_type, DoubleTensorType):
        return "tensor(double)"
    if isinstance(data_type, StringTensorType):
        return "tensor(string)"
    if isinstance(data_type, Int64TensorType):
        return "tensor(int64)"
    if isinstance(data_type, Int32TensorType):
        return "tensor(int32)"
    if isinstance(data_type, BooleanTensorType):
        return "tensor(bool)"
    raise NotImplementedError(
        "Unsupported data_type '{}'. You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "".format(data_type)
    )


def _guess_numpy_type(data_type, dims):
    if data_type == np.float32:
        return FloatTensorType(dims)
    if data_type == np.float64:
        return DoubleTensorType(dims)
    if (
        data_type in (np.str_, str, object)
        or str(data_type) in ("<U1",)
        or (hasattr(data_type, "type") and data_type.type is np.str_)
    ):
        return StringTensorType(dims)
    if data_type in (np.int64,) or str(data_type) == "<U6":
        return Int64TensorType(dims)
    if data_type in (np.int32,) or str(data_type) in ("<U4",):
        return Int32TensorType(dims)
    if data_type == np.uint8:
        return UInt8TensorType(dims)
    if data_type in (np.bool_, bool):
        return BooleanTensorType(dims)
    if data_type in (np.str_, str):
        return StringTensorType(dims)
    if data_type == np.int8:
        return Int8TensorType(dims)
    if data_type == np.int16:
        return Int16TensorType(dims)
    if data_type == np.uint64:
        return UInt64TensorType(dims)
    if data_type == np.uint32:
        return UInt32TensorType(dims)
    if data_type == np.uint16:
        return UInt16TensorType(dims)
    if data_type == np.float16:
        return Float16TensorType(dims)
    if Complex64TensorType is not None:
        if data_type == np.complex64:
            return Complex64TensorType(dims)
        if data_type == np.complex128:
            return Complex128TensorType(dims)
    raise NotImplementedError(
        "Unsupported data_type %r (type=%r). You may raise an issue "
        "at https://github.com/onnx/sklearn-onnx/issues."
        "" % (data_type, type(data_type))
    )


def guess_data_type(type_, shape=None):
    """
    Guess the datatype given the type type_
    """
    if isinstance(type_, onnx.TensorProto):
        return _guess_type_proto(type, shape)
    if isinstance(type_, str):
        return _guess_type_proto_str(type_, shape)
    if hasattr(type_, "columns") and hasattr(type_, "dtypes"):
        # DataFrame
        return [
            (name, _guess_numpy_type(dt, [None, 1]))
            for name, dt in zip(type_.columns, type_.dtypes)
        ]
    if hasattr(type_, "name") and hasattr(type_, "dtype"):
        # Series
        return [(type_.name, _guess_numpy_type(type_.dtype, [None, 1]))]
    if hasattr(type_, "shape") and hasattr(type_, "dtype"):
        # array
        return [("input", _guess_numpy_type(type_.dtype, type_.shape))]
    raise TypeError(
        "Type {} cannot be converted into a "
        "DataType. You may raise an issue at "
        "https://github.com/onnx/sklearn-onnx/issues."
        "".format(type(type_))
    )


def guess_numpy_type(data_type):
    """
    Guess the corresponding numpy type based on data_type.
    """
    if data_type in (
        np.float64,
        np.float32,
        np.int8,
        np.uint8,
        np.str_,
        np.bool_,
        np.int32,
        np.int64,
    ):
        return data_type
    if data_type == str:  # noqa: E721
        return np.str_
    if data_type == bool:  # noqa: E721
        return np.bool_
    if isinstance(data_type, FloatTensorType):
        return np.float32
    if isinstance(data_type, DoubleTensorType):
        return np.float64
    if isinstance(data_type, Int32TensorType):
        return np.int32
    if isinstance(data_type, Int64TensorType):
        return np.int64
    if isinstance(data_type, StringTensorType):
        return np.str_
    if isinstance(data_type, BooleanTensorType):
        return np.bool_
    if Complex64TensorType is not None:
        if data_type in (np.complex64, np.complex128):
            return data_type
        if isinstance(data_type, Complex64TensorType):
            return np.complex64
        if isinstance(data_type, Complex128TensorType):
            return np.complex128
    raise NotImplementedError("Unsupported data_type '{}'.".format(data_type))


def guess_proto_type(data_type):
    """
    Guess the corresponding proto type based on data_type.
    """
    if isinstance(data_type, FloatTensorType):
        return onnx.TensorProto.FLOAT
    if isinstance(data_type, DoubleTensorType):
        return onnx.TensorProto.DOUBLE
    if isinstance(data_type, Int32TensorType):
        return onnx.TensorProto.INT32
    if isinstance(data_type, Int64TensorType):
        return onnx.TensorProto.INT64
    if isinstance(data_type, StringTensorType):
        return onnx.TensorProto.STRING
    if isinstance(data_type, BooleanTensorType):
        return onnx.TensorProto.BOOL
    if isinstance(data_type, Int8TensorType):
        return onnx.TensorProto.INT8
    if isinstance(data_type, UInt8TensorType):
        return onnx.TensorProto.UINT8
    if Complex64TensorType is not None:
        if isinstance(data_type, Complex64TensorType):
            return onnx.TensorProto.COMPLEX64
        if isinstance(data_type, Complex128TensorType):
            return onnx.TensorProto.COMPLEX128
    raise NotImplementedError("Unsupported data_type '{}'.".format(data_type))


def guess_tensor_type(data_type):
    """
    Guess the corresponding variable output type based
    on input type. It returns type if *data_type* is a real.
    It returns *FloatTensorType* if *data_type* is an integer.
    """
    if isinstance(data_type, DoubleTensorType):
        return DoubleTensorType()
    if isinstance(data_type, DictionaryType):
        return guess_tensor_type(data_type.value_type)
    if Complex64TensorType is not None and isinstance(
        data_type, (Complex64TensorType, Complex128TensorType)
    ):
        return data_type.__class__()
    if not isinstance(
        data_type,
        (
            Int64TensorType,
            Int32TensorType,
            BooleanTensorType,
            FloatTensorType,
            StringTensorType,
            DoubleTensorType,
            Int8TensorType,
            UInt8TensorType,
        ),
    ):
        raise TypeError(
            "data_type is not a tensor type but '{}'.".format(type(data_type))
        )
    return FloatTensorType()
