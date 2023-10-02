# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for Spark ML to Onnx conversion intended for the end user mainly
"""
from ..common.data_types import StringTensorType, FloatTensorType


def buildInitialTypesSimple(dataframe):
    types = []
    for field in dataframe.schema.fields:
        types.append((field.name, getTensorTypeFromSpark(str(field.dataType))))
    return types


def getTensorTypeFromSpark(sparktype):
    if sparktype in ("StringType", "StringType()"):
        return StringTensorType([1, 1])
    if (
        sparktype == "DecimalType"
        or sparktype == "DecimalType()"
        or sparktype == "DoubleType"
        or sparktype == "DoubleType()"
        or sparktype == "FloatType"
        or sparktype == "FloatType()"
        or sparktype == "LongType"
        or sparktype == "LongType()"
        or sparktype == "IntegerType"
        or sparktype == "IntegerType()"
        or sparktype == "ShortType"
        or sparktype == "ShortType()"
        or sparktype == "ByteType"
        or sparktype == "ByteType()"
        or sparktype == "BooleanType"
        or sparktype == "BooleanType()"
    ):
        return FloatTensorType([None, 1])
    raise TypeError(f"Cannot map this type to Onnx types: {sparktype}.")


def buildInputDictSimple(dataframe):
    import numpy

    result = {}
    for field in dataframe.schema.fields:
        if str(field.dataType) in ("StringType", "StringType()"):
            result[field.name] = dataframe.select(field.name).toPandas().values
        else:
            result[field.name] = (
                dataframe.select(field.name).toPandas().values.astype(numpy.float32)
            )
    return result


class SparkMlConversionError(Exception):
    pass
