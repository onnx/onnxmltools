# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import coremltools
from uuid import uuid4
from ...proto import onnx
from ...proto import onnx_proto
from ..common._topology import convert_topology
from ._parse import parse_coreml

# Import modules to invoke function registrations
from . import operator_converters
from . import shape_calculators
from .operator_converters import neural_network as nn_converters
from .shape_calculators import neural_network as nn_shape_calculators


def convert(model, name=None, initial_types=None, doc_string='',
            targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    '''
    This function converts the specified CoreML model into its ONNX counterpart. Some information such as the produced
    ONNX model name can be specified.
    :param model: A CoreML model (https://apple.github.io/coremltools/coremlspecification/sections/Model.html#model) or
    a CoreML MLModel object
    :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
    name and a type defined in data_types.py.
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: A string attached onto the produced ONNX model
    :param targeted_onnx: A string (for example, '1.1.2' and '1.2') used to specify the targeted ONNX version of the
    produced model. If ONNXMLTools cannot find a compatible ONNX python package, an error may be thrown.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: An ONNX model (type: ModelProto) which is equivalent to the input CoreML model

    Example of initial types:
    Assume that 'A' and 'B' are two root variable names used in the CoreML model you want to convert. We can specify
    their types via
    >>> from onnxmltools.convert.common.data_types import FloatTensorType
    >>> initial_type = [('A', FloatTensorType([40, 12, 1, 1])), ('B', FloatTensorType([1, 32, 1, 1]))]
    '''
    if isinstance(model, coremltools.models.MLModel):
        spec = model.get_spec()
    else:
        spec = model

    if name is None:
        name = str(uuid4().hex)

    # Parse CoreML model as our internal data structure (i.e., Topology)
    topology = parse_coreml(spec, initial_types, targeted_onnx, custom_conversion_functions, custom_shape_calculators)

    # Parse CoreML description, author, and license. Those information will be attached to the final ONNX model.
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
            metadata_props.append(entry)

    # Convert our Topology object into ONNX. The outcome is an ONNX model.
    onnx_model = convert_topology(topology, name, doc_string, targeted_onnx)

    # Edit ONNX model's attributes related to CoreML's meta information
    if len(metadata_props) > 0:
        onnx_model.metadata_props.extend(metadata_props)

    return onnx_model
