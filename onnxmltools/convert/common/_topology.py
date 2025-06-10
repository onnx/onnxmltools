# SPDX-License-Identifier: Apache-2.0

import warnings
import onnx
from onnx import helper
from ._registration import get_converter
from .data_types import StringType, FloatType, Int64Type, TensorType
from ._container import ModelComponentContainer
from .onnx_ex import get_maximum_opset_supported, OPSET_TO_IR_VERSION
from .utils import get_model_version, get_domain, get_producer_version, get_producer


KNOWN_METADATA_PROPS = {
    "Image.BitmapPixelFormat": ["gray8", "rgb8", "bgr8", "rgba8", "bgra8"],
    "Image.ColorSpaceGamma": ["linear", "srgb"],
    "Image.NominalPixelRange": [
        "nominalrange_0_255",
        "normalized_0_1",
        "normalized_1_1",
        "nominalrange_16_235",
    ],
}


def _get_main_opset_version(model):
    """
    Returns the main opset version.
    """
    for op in model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            return op.version
    return None


def _validate_metadata(metadata_props):
    """
    Validate metadata properties and possibly show warnings or throw exceptions.

    :param metadata_props: A dictionary of metadata properties,
    with property names and values (see :func:`~onnxmltools.utils.metadata_props.add_metadata_props` for examples)
    """
    if len(metadata_props) != len(metadata_props):
        raise RuntimeError("Duplicate metadata props found")

    for key, value in metadata_props.items():
        valid_values = KNOWN_METADATA_PROPS.get(key)
        if valid_values and value.lower() not in valid_values:
            warnings.warn(
                "Key {} has invalid value {}. Valid values are {}".format(
                    key, value, valid_values
                )
            )


def add_metadata_props(onnx_model, metadata_props, target_opset):
    """
    Add metadata properties to the model. See recommended key names at:
    `Extensibility -
        Metadata <https://github.com/onnx/onnx/blob/296953db87b79c0137c5d9c1a8f26dfaa2495afc/docs/IR.md#metadata>`_ and
    `Optional Metadata <https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-metadata>`_


    :param onnx_model: ONNX model object
    :param metadata_props: A dictionary of metadata properties,
        with property names and values (example: `{ 'model_author': 'Alice', 'model_license': 'MIT' }`)
    :param target_opset: Target ONNX opset
    """
    if target_opset < 7:
        warnings.warn(
            "Metadata properties are not supported in targeted opset - %d"
            % target_opset
        )
        return
    _validate_metadata(metadata_props)
    new_metadata = {x.key: x.value for x in onnx_model.metadata_props}
    new_metadata.update(metadata_props)
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(
        onnx.StringStringEntryProto(key=key, value=value)
        for key, value in metadata_props.items()
    )


def make_model_ex(
    graph, imported_opset_pairs, target_default_opset, metadata_props=None, **kwargs
):
    onnx_model = helper.make_model(graph, **kwargs)

    # Merge operator sets for the same domain, the largest version number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in imported_opset_pairs:
        if op_domain not in purified_operator_set:
            if op_domain == "":
                # Initializers are a subset of graph inputs for IR_VERSION <= 3 (target opset < 8).
                # Need upgrade opv since initializers are separate for IR_VERSION >= 4 to pass onnx.checker.
                if (
                    op_version < 8
                    and target_default_opset is not None
                    and target_default_opset >= 8
                ):
                    op_version = 8
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(
                purified_operator_set[op_domain], op_version
            )

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by helper.make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
        i += 1
        if op_domain == "" or op_domain == "ai.onnx":
            if target_default_opset < op_version:
                raise RuntimeError(
                    (
                        "The specified opset %d is too low to convert this model, "
                        + "which requires at least opset %d."
                    )
                    % (target_default_opset, op_version)
                )
            else:
                pass

    # Add extra information
    if metadata_props:
        add_metadata_props(onnx_model, metadata_props, target_default_opset)
    opv = _get_main_opset_version(onnx_model) or target_default_opset
    irv = OPSET_TO_IR_VERSION.get(opv, onnx.IR_VERSION)
    onnx_model.ir_version = irv
    onnx_model.producer_name = (
        kwargs.get("producer_name") if "producer_name" in kwargs else get_producer()
    )
    onnx_model.producer_version = get_producer_version()
    onnx_model.domain = kwargs.get("domain") if "domain" in kwargs else get_domain()
    onnx_model.model_version = get_model_version()
    return onnx_model


def _build_onnx_model(node_list):
    regenerated = []
    for n_ in node_list:
        nodes = n_.generate()
        regenerated.extend(nodes)
    return regenerated


def _visit(name_to_node_map, n_name, result):
    node = name_to_node_map[n_name]
    if node.status == "perm":
        return
    if node.status == "temp":
        raise Exception("This graph is not a DAG")
    node.status = "temp"
    for m in node.successor:
        if m.origin is not None:
            _visit(name_to_node_map, m.unique_name, result)
    node.status = "perm"
    result.insert(0, node.idx)


def _topological_sort(node_list):
    name_to_node_map = dict()

    def _get_unmark_node(name_to_node_map):
        for k, v in name_to_node_map.items():
            if v.status == "unmark":
                return k
        return None

    result = []
    name_set = set()
    for idx_, n_ in enumerate(node_list):
        setattr(n_, "idx", idx_)

    for n_ in node_list:
        name = n_.unique_name
        name_set.add(name)
        setattr(n_, "status", "unmark")
        name_to_node_map.update({name: n_})

    n_name = _get_unmark_node(name_to_node_map)
    while n_name:
        _visit(name_to_node_map, n_name, result)
        n_name = _get_unmark_node(name_to_node_map)

    result_nodes = [node_list[result[idx]] for idx in range(len(node_list))]
    return result_nodes


def convert_topology(
    topology, model_name, doc_string, target_opset, channel_first_inputs=None
):
    """
    This function is used to convert our Topology object defined in _parser.py into a ONNX model (type: ModelProto).
    :param topology: The Topology object we are going to convert
    :param model_name: GraphProto's name. Let "model" denote the returned model. The string "model_name" would be
    assigned to "model.graph.name."
    :param doc_string: A string attached to the produced model
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :return: a ONNX ModelProto
    """
    opset_from_onnx_version = get_maximum_opset_supported()
    if target_opset is None:
        target_opset = opset_from_onnx_version
    elif target_opset > opset_from_onnx_version:
        raise RuntimeError(
            (
                "target_opset %d is higher than the number of the installed onnx package"
                + " or the converter support (%d)."
            )
            % (target_opset, opset_from_onnx_version)
        )

    topology._initialize_graph_status_for_traversing()

    container = ModelComponentContainer(target_opset)

    # Put roots and leaves as ONNX's model into buffers. They will be added into ModelComponentContainer later.
    tensor_inputs = {}
    other_inputs = {}
    tensor_outputs = {}
    other_outputs = {}
    for scope in topology.scopes:
        for variable in scope.variables.values():
            if variable.is_root:
                if isinstance(
                    variable.type, (TensorType, Int64Type, FloatType, StringType)
                ):
                    tensor_inputs[variable.raw_name] = variable
                else:
                    other_inputs[variable.raw_name] = variable
            if variable.is_leaf:
                if isinstance(
                    variable.type, (TensorType, Int64Type, FloatType, StringType)
                ):
                    tensor_outputs[variable.raw_name] = variable
                else:
                    other_outputs[variable.raw_name] = variable

    # Add roots the graph according to their order in the original model
    invalid_name = []
    nhwc_inputs = []
    if channel_first_inputs is None:
        channel_first_inputs = []
    for name in topology.raw_model.input_names:
        # Check input naming convention
        input_name = name.replace("_", "").replace(":", "").replace("/", "")
        if input_name and (input_name[0].isdigit() or (not input_name.isalnum())):
            invalid_name.append(name)
        if name in tensor_inputs:
            onnx_input = tensor_inputs[name]  # type: Variable
            if name in channel_first_inputs or (
                name.endswith(":0") and name[:-2] in channel_first_inputs
            ):
                nhwc_inputs.append(onnx_input.full_name)
                s = onnx_input.type.shape
                onnx_input.type.shape = [s[0], s[3], s[1], s[2]]
            container.add_input(onnx_input)

    if invalid_name:
        warnings.warn(
            "Some input names are not compliant with ONNX naming convention: %s"
            % invalid_name
        )
    for name in topology.raw_model.input_names:
        if name in other_inputs:
            container.add_input(other_inputs[name])

    # Add leaves the graph according to their order in the original model
    invalid_name = []
    for name in topology.raw_model.output_names:
        # Check output naming convention
        output_name = name.replace("_", "").replace(":", "").replace("/", "")
        if output_name and (output_name[0].isdigit() or (not output_name.isalnum())):
            invalid_name.append(name)
        if name in tensor_outputs:
            container.add_output(tensor_outputs[name])
    if invalid_name:
        warnings.warn(
            "Some output names are not compliant with ONNX naming convention: %s"
            % invalid_name
        )
    for name in topology.raw_model.output_names:
        if name in other_outputs:
            container.add_output(other_outputs[name])

    # Traverse the graph from roots to leaves
    for operator in topology.topological_operator_iterator():
        scope = next(scope for scope in topology.scopes if scope.name == operator.scope)
        if operator.type in topology.custom_conversion_functions:
            topology.custom_conversion_functions[operator.type](
                scope, operator, container
            )
        else:
            # Convert the selected operator into some ONNX objects and save them into the container
            get_converter(operator.type)(scope, operator, container)

    # When calling ModelComponentContainer's add_initializer(...), nothing is added into the input list.
    # However, for ONNX target opset < 9, initializers should also be model's (GraphProto) inputs.
    # Thus, we create ValueInfoProto objects from initializers (type: TensorProto) directly and
    # then add them into model's input list.
    extra_inputs = []  # ValueInfoProto list of the initializers
    for tensor in container.initializers:
        # Sometimes (especially when creating optional input values such as RNN's initial hidden state), an initializer
        # is also one of the original model's input, so it has been added into the container's input list. If this is
        # the case, we need to skip one iteration to avoid duplicated inputs.
        if tensor.name in [value_info.name for value_info in container.inputs]:
            continue

        # Initializers are always tensors so we can just call make_tensor_value_info(...)
        value_info = helper.make_tensor_value_info(
            tensor.name, tensor.data_type, tensor.dims
        )
        extra_inputs.append(value_info)

    # enable the ONNX optimizations
    if container.enable_optimizer:
        from onnxconverter_common.optimizer import optimize_onnx

        nodes = optimize_onnx(
            container.nodes,
            nhwc_inputs,
            container.inputs + extra_inputs,
            container.outputs,
        )
    else:
        nodes = container.nodes

    # Create a graph from its main components
    if container.target_opset < 9:
        # Before ONNX opset 9, initializers need to be passed in with inputs
        graph = helper.make_graph(
            nodes,
            model_name,
            container.inputs + extra_inputs,
            container.outputs,
            container.initializers,
        )
    else:
        # In ONNX opset 9 and above, initializers are included as operator
        # inputs, and therefore do not need to be passed as extra_inputs
        graph = helper.make_graph(
            nodes,
            model_name,
            container.inputs,
            container.outputs,
            container.initializers,
        )

    # Add extra information related to the graph
    graph.value_info.extend(container.value_info)
    onnx_model = make_model_ex(
        graph,
        container.node_domain_version_pair_sets,
        target_opset,
        topology.metadata_props,
        doc_string=doc_string,
    )
    return onnx_model
