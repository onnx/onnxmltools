# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from ...proto import onnx
from ..common._topology import convert_topology
from ._parse import parse_sparkml
from . import operator_converters


def convert(model, name=None, initial_types=None, doc_string='', target_opset=None,
            targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None,
            spark_session=None):
    '''
    This function produces an equivalent ONNX model of the given spark-ml model. The supported spark-ml
    modules are listed below.

    * Preprocessings and transformations:
      1.  pyspark.ml.feature.DictVectorizer
      3.  preprocessing.LabelEncoder
      5.  preprocessing.OneHotEncoder

    * Linear classification and regression:
      9.  pyspark.ml.classification.LinearRegression

    * Support vector machine for classification and regression

    * Tree-based models for classification and regression

    * pipeline
      29. pipeline.Pipeline

    For pipeline conversion, user needs to make sure each component is one of our supported items (1)-(24).

    This function converts the specified spark-ml model into its ONNX counterpart. Notice that for all conversions,
    initial types are required.  ONNX model name can also be specified.

    :param model: A spark-ml model
    :param initial_types: a python list. Each element is a tuple of a variable name and a type defined in data_types.py
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
    produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input spark-ml model

    Example of initial_types:
    Assume that the specified spark-ml model takes a heterogeneous list as its input. If the first 5 elements are
    floats and the last 10 elements are integers, we need to specify initial types as below. The [1] in [1, 5] indicates
    the batch size here is 1.
    >>> from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
    >>> initial_type = [('float_input', FloatTensorType([1, 5])), ('int64_input', Int64TensorType([1, 10]))]
    '''
    if initial_types is None:
        raise ValueError('Initial types are required. See usage of convert(...) in \
                         onnxmltools.convert.sparkml.convert for details')

    if name is None:
        name = str(uuid4().hex)

    target_opset = target_opset if target_opset else get_maximum_opset_supported()
    # Parse spark-ml model as our internal data structure (i.e., Topology)
    topology = parse_sparkml(spark_session, model, initial_types, target_opset, custom_conversion_functions, custom_shape_calculators)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string, target_opset, targeted_onnx)

    return onnx_model
