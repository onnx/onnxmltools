# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ..common._topology import convert_topology
from ._parse import parse_keras

# Register conversion functions and shape inference functions
from . import operator_converters
from . import shape_calculators


def convert(model, name=None, initial_types=None, doc_string=''):
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
    '''
    topology = parse_keras(model, initial_types)

    topology.compile()

    if name is None:
        name = str(uuid4().hex)

    onnx_model = convert_topology(topology, name, doc_string)

    return onnx_model
