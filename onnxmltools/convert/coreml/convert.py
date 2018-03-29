# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import coremltools
from uuid import uuid4
from ...proto import onnx_proto
from ..common import utils
from ..common._topology import convert_topology
from ._parse import parse_coreml


def convert(model, name=None, initial_types=None, doc_string=''):
    '''
    This function converts the specified CoreML model into its ONNX counterpart. Some information such as the produced
    ONNX model name can be specified.
    :param model: A CoreML model (https://apple.github.io/coremltools/coremlspecification/sections/Model.html#model) or
    a CoreML MLModel object
    :param initial_types: a python dictionary. Its keys are variable name while the corresponding values are their types
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :return: An ONNX model (type: ModelProto) which is equivalent to the input CoreML model

    Example of initial types:
    Assume that 'A' and 'B' are two root variable names used in the CoreML model you want to convert. We can specify
    their types via
    >>> from _data_types import FloatTensorType
    >>> initial_type = {'A': FloatTensorType([40, 12, 1, 1]), 'B': FloatTensorType([1, 32, 1, 1])}
    '''
    if isinstance(model, coremltools.models.MLModel):
        spec = model.get_spec()
    else:
        spec = model

    if name is None:
        name = str(uuid4().hex)

    if initial_types is None:
        initial_types = dict()

    # Parse CoreML model as our internal data structure (i.e., Topology)
    topology = parse_coreml(spec, initial_types)

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name)

    # Parse CoreML description, author, and license
    metadata = spec.description.metadata
    metadata_props = []
    if metadata:
        if not doc_string and metadata.shortDescription:
            doc_string = metadata.shortDescription  # If doc_string is not specified, we use description from CoreML
        if metadata.author:
            entry = onnx_proto.StringStringEntryProto()
            entry.key = 'author'
            entry.value = metadata.author
            metadata_props.append(entry)
        if metadata.license:
            entry = onnx_proto.StringStringEntryProto()
            entry.key = 'license'
            entry.value = metadata.license

    # Specify ONNX model's attributes which are not directly related to computational graph
    if len(metadata_props) > 0:
        onnx_model.metadata_props.extend(metadata_props)
    onnx_model.ir_version = onnx_proto.IR_VERSION
    onnx_model.producer_name = utils.get_producer()
    onnx_model.producer_version = utils.get_producer_version()
    onnx_model.domain = utils.get_domain()
    onnx_model.model_version = utils.get_model_version()
    onnx_model.doc_string = doc_string

    return onnx_model
