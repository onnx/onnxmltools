# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import coremltools
from uuid import uuid4
from ...proto import onnx_proto
from ..common._topology import convert_topology
from ._parse import parse_coreml

# Import modules to invoke function registrations
from . import operator_converters
from . import shape_calculators
from .operator_converters import neural_network as nn_converters
from .shape_calculators import neural_network as nn_shape_calculators


# The conversion of a CoreLM model consists of several major steps.
#   1. Put the input model into a Container object (i.e., onnxmltools.convert.common._container.CoremlModelContainer)
#   2. Create a empty Topology (i.e., onnxmltools.convert.common._topology.Topology) object. It's an an abstract
#      computational graph of the input model.
#   3. Parse the CoreML model as a computational graph. We may add variables and operators into the Topology object
#      defined in the previous step.
#   4. Call the member function, compile, in Topology. There are two important steps.
#        a. First, we may feed input variables specified in the Container (defined in Step 1) into our computational
#           graph. Then, we may evaluate the computational graph. Unreachable variables and operators would be removed.
#        b. Second, we may invoke the shape calculators for all existing operators in a topological order.
#   5. Finally, the conversion functions of all existing operators are called in a topological order.
#
#   Note that steps 1-4 can be found in onnxmltools.convert.coreml._parse.parse_coreml. Step 5 is implemented in
#   onnxmltools.common._topology.convert_topology. Step 4-a is onnxmltools.convert.common._topology.Topology._prune and
#   step 4-b is onnxmltools.convert.common._topology.Topology._infer_all_types.
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
    >>> from onnxmltools.convert.common.data_types import FloatTensorType
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
    onnx_model = convert_topology(topology, name, doc_string)

    # Edit ONNX model's attributes related to CoreML's meta information
    if len(metadata_props) > 0:
        onnx_model.metadata_props.extend(metadata_props)

    return onnx_model
