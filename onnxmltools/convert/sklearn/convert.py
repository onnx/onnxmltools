# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ...proto import onnx_proto
from ..common import model_util
from ._parse import parse_sklearn
from ..coreml._topology import convert_topology
from . import shape_calculators
from . import operator_converters

def convert(model, name=None, initial_types=[], doc_string=''):
    '''
    This function converts the specified scikit-learn model into its ONNX counterpart. Notice that for all conversions,
    initial types are required.  ONNX model name can also be specified.
    :param model: A scikit-learn model
    :param initial_types: a python list whose elements are data types defined in _data_types.py 
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :return: An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model

    Example of initial types:
    Assume that the specified scikit-learn model takes a heterogeneous list as its input. If the first 5 elements are
    floats and the last 10 elements are integers, we need to specify initial types as below. The [1] in [1, 5] indicates
    the batch size here is 1.
    >>> from _data_types import FloatTensorType
    >>> initial_type = [FloatTensorType([1, 5]), Int64TensorType([1, 10])]
    '''
    if name is None:
        name = str(uuid4().hex)

    # Parse scikit-learn model as our internal data structure (i.e., Topology)
    topology = parse_sklearn(model, initial_types)

    # Infer variable shapes
    topology.compile()

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name)

    # Add extra information
    onnx_model.ir_version = onnx_proto.IR_VERSION
    onnx_model.producer_name = model_util.get_producer()
    onnx_model.producer_version = model_util.get_producer_version()
    onnx_model.domain = model_util.get_domain()
    onnx_model.model_version = model_util.get_model_version()
    onnx_model.doc_string = doc_string

    return onnx_model
