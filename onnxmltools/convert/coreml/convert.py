import coremltools
from uuid import uuid4
from . import _converters
from . import _parser
from ... import __version__
from ...proto import onnx_proto
from ..common import model_util

def convert(model, name=None, initial_types={}, doc_string=''):
    if isinstance(model, coremltools.models.MLModel):
        spec = model.get_spec()
    else:
        spec = model

    if name is None:
        name = str(uuid4().hex)

    topology = _parser.parse_coreml(spec, initial_types)
    # Uncomment this line to visualize the intermediate graph for debugging
    #_parser.visualize_topology(topology, filename=name, view=True)
    # Construct the parts of a ONNX model related to computational graph
    onnx_model = _converters.convert_topology(topology, name)

    # Convert CoreML description, author and license
    metadata = spec.description.metadata
    metadata_props = []
    if metadata:
        if not doc_string and metadata.shortDescription:
            doc_string = metadata.shortDescription # If doc_string is not specified, we use description from CoreML
        if metadata.author:
            metadata_props.append(model_util.make_string_string_entry('author', metadata.author))
        if metadata.license:
            metadata_props.append(model_util.make_string_string_entry('license', metadata.license))

    # Specify ONNX model's attributes which are not directly related to computational graph
    if len(metadata_props) > 0:
        onnx_model.metadata_props.extend(metadata_props)
    onnx_model.ir_version = onnx_proto.IR_VERSION
    onnx_model.producer_name = model_util.get_producer()
    onnx_model.producer_version = model_util.get_producer_version()
    onnx_model.domain = model_util.get_domain()
    onnx_model.model_version = model_util.get_model_version()
    onnx_model.doc_string = doc_string

    return onnx_model

def make_model(name, ir_version, producer, producer_version, domain, model_version, doc_string, metadata_props,
               operator_domain_version_pairs, nodes, inputs, outputs, values, initializer=list()):
    model = onnx_proto.ModelProto()
    model.ir_version = ir_version
    model.producer_name = producer
    model.producer_version = producer_version
    model.domain = domain
    model.model_version = model_version
    model.doc_string = doc_string
    if len(metadata_props) > 0:
        model.metadata_props.extend(metadata_props)
    for op_domain, op_version in operator_domain_version_pairs:
        op_set = model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
    graph = model.graph
    graph.name = name
    graph.node.extend(nodes)
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.value_info.extend(values)
    graph.initializer.extend(initializer)
    return model
    return onnx_model


