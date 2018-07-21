# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ...proto import onnx
from ..common._topology import convert_topology
from ..common._registration import register_converter
from ..common._registration import register_shape_calculator
from ._parse import parse_keras

# Register conversion functions and shape inference functions
from . import operator_converters
from . import shape_calculators


def convert(model, name=None, initial_types=None, doc_string='',
            targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    '''
    Convert Keras-Tensorflow Model and Sequence objects into Topology. Note that default batch size is 1 here instead of
    `None` used in CoreML conversion framework. To overwrite this behavior, we can specify initial_types. Assume that a
    Keras tensor is named input:0 and its shape is [None, 3]. If the desired batch size is 10, we can specify
    >>> from onnxmltools.convert.common.data_types import FloatTensorType
    >>> initial_types=[('input:0', FloatTensorType([10, 3]))]

    :param model: A Keras model (Model or Sequence object)
    :param name: Optional graph name of the produced ONNX model
    :param initial_types: A list providing types for some input variables. Each element is a tuple of a variable name
    and a type defined in data_types.py.
    :param doc_string: A string attached onto the produced ONNX model
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
    produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input Keras model
    '''

    topology = parse_keras(model, initial_types, targeted_onnx, custom_conversion_functions, custom_shape_calculators)

    topology.compile()

    if name is None:
        name = str(uuid4().hex)

    onnx_model = convert_topology(topology, name, doc_string, targeted_onnx)
    return onnx_model
