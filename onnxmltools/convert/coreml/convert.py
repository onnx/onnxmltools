#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import coremltools
from ...proto import onnx_proto
from ..common import registration
from ..common import ModelBuilder
from ..common import NodeBuilder
from ..common import model_util
from .CoremlConvertContext import CoremlConvertContext as ConvertContext
from .datatype import make_value_info
# These are not referenced directly but are imported to initialize the registration call
from . import ArrayFeatureExtractorConverter
from . import DictVectorizerConverter
from . import FeatureVectorizerConverter
from . import GLMClassifierConverter
from . import GLMRegressorConverter
from . import IdentityConverter
from . import ImputerConverter
from . import NormalizerConverter
from . import OneHotEncoderConverter
from . import ScalerConverter
from . import SupportVectorClassifierConverter
from . import SupportVectorRegressorConverter
from . import TreeEnsembleClassifierConverter
from . import TreeEnsembleRegressorConverter
from . import NeuralNetwork


def convert(model, name=None, doc_string=''):
    '''
    This function converts the specified CoreML model into its ONNX counterpart. Some information such as the produced
    ONNX model name can be specified.
    :param model: A CoreML model (https://apple.github.io/coremltools/coremlspecification/sections/Model.html#model) or
    a CoreML MLModel object
    :param name: The name of the graph (type: GraphProto) in the produced ONNX model (type: ModelProto)
    :param doc_string: (optional) Override existing CoreML model shortDescription
    :return: A ONNX model (type: ModelProto) which is equivalent to the input CoreML model
    '''
    if isinstance(model, coremltools.models.MLModel):
        spec = model.get_spec()
    else:
        spec = model

    # Create name management object
    context = ConvertContext()

    # Determine default batch size
    model_type = spec.WhichOneof('Type') 
    batch_size = 1
    if model_type in ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork']:
        batch_size = 'None' # Use variable-length tensor as input for neural network

    # Save top-level inputs
    inputs = []
    for graph_input in spec.description.input:
        input_tensor = make_value_info(graph_input, batch_size)
        inputs.append(input_tensor)
    context.extend_top_level_inputs(inputs)

    # Save the ONNX names of all inputs because some following steps may over-write them
    reserved_input_names = []
    for input in inputs:
        reserved_input_names.append(context.get_onnx_name(input.name))

    # Save top-level outputs
    outputs = []
    for graph_output in spec.description.output:
        output_tensor = make_value_info(graph_output, batch_size)
        outputs.append(output_tensor)
    context.extend_top_level_outputs(outputs)

    # Convert CoreML components to ONNX nodes
    nodes = _convert_coreml_node(context, spec)

    # Replace CoreML name with ONNX name
    for input, onnx_name in zip(inputs, reserved_input_names):
        input.name = onnx_name

    # Replace CoreML name with ONNX name
    for output in outputs:
        output.name = context.get_onnx_name(output.name)

    # Convert CoreML description, author and license
    metadata = spec.description.metadata
    metadata_props = []
    if metadata:
        if not doc_string and metadata.shortDescription:
            doc_string = metadata.shortDescription
        if metadata.author:
            metadata_props.append(model_util.make_string_string_entry('author', metadata.author))
        if metadata.license:
            metadata_props.append(model_util.make_string_string_entry('license', metadata.license))

    mb = ModelBuilder(name, doc_string, metadata_props)
    mb.add_inputs(inputs)
    mb.add_outputs(outputs)
    for node in nodes:
        mb.add_nodes([node.onnx_node])
        mb.add_initializers(node.initializers)
        mb.add_values(node.values)
        mb.add_op_set(node.op_set)

    return mb.make_model()


def _do_convert(context, converter, cm_node, input=None, output=None):
    converter.validate(cm_node)
    if input is None and output is None:
        nodes = converter.convert(context, cm_node)
    else:
        nodes = converter.convert(context, cm_node, input, output)

    if isinstance(nodes, list):
        return nodes
    return [nodes]


def _create_post_processing_nodes(context, coreml_nn, default_proba_tensor_name,
                                  reserved_label_name, reserved_proba_dict_name):
    nodes = []
    coreml_label_name = coreml_nn.description.predictedFeatureName
    coreml_proba_name = coreml_nn.description.predictedProbabilitiesName

    if coreml_proba_name == '':
        proba_tensor_name = default_proba_tensor_name
    else:
        proba_tensor_name = context.get_onnx_name(coreml_proba_name)

    # Load class labels
    label_loader_builder = NodeBuilder(context, 'Constant')
    label_buf_name = context.get_unique_name('ClassLabelBuffer')
    label_loader_builder.add_output(label_buf_name)
    if coreml_nn.neuralNetworkClassifier.WhichOneof('ClassLabels') == 'stringClassLabels':
        labels = coreml_nn.neuralNetworkClassifier.stringClassLabels.vector
        label_tensor = model_util.make_tensor('Content', onnx_proto.TensorProto.STRING, [
            len(labels)], [s.encode('ascii') for s in labels])
    else:
        labels = coreml_nn.neuralNetworkClassifier.int64ClassLabels.vector
        label_tensor = model_util.make_tensor(
            'Content', onnx_proto.TensorProto.INT64, [len(labels)], labels)
    label_loader_builder.add_attribute('value', label_tensor)
    nodes.append(label_loader_builder.make_node())

    # Deduce best label's index from probability tensor
    extracted_id = context.get_unique_name('BestLabelId')
    id_extractor_builder = NodeBuilder(context, 'ArgMax')
    id_extractor_builder.add_input(proba_tensor_name)
    id_extractor_builder.add_output(extracted_id)
    id_extractor_builder.add_attribute('axis', 1)
    id_extractor_builder.add_attribute('keepdims', 1)
    nodes.append(id_extractor_builder.make_node())

    # Extract the best label
    label_extractor_builder = NodeBuilder(context, 'ArrayFeatureExtractor', op_domain='ai.onnx.ml')
    label_extractor_builder.add_input(label_buf_name)
    label_extractor_builder.add_input(extracted_id)
    label_extractor_builder.add_output(reserved_label_name)
    nodes.append(label_extractor_builder.make_node())

    # Create probability tensor to probability map
    label_type = None
    proba_type = None
    for o in coreml_nn.description.output:
        if o.name == coreml_label_name:
            label_type = o.type.WhichOneof('Type')
        if o.name == coreml_proba_name:
            proba_type = o.type.WhichOneof('Type')
    if coreml_proba_name != '':
        map_constructor_builder = NodeBuilder(context, 'ZipMap', op_domain='ai.onnx.ml')
        map_constructor_builder.add_input(proba_tensor_name)
        map_constructor_builder.add_output(reserved_proba_dict_name)
        if label_type == 'stringType':
            attr_name = 'classlabels_strings'
            attr_content = [s.encode('ascii') for s in coreml_nn.neuralNetworkClassifier.stringClassLabels.vector]
        else:
            attr_name = 'classlabels_int64s'
            attr_content = list(int(i) for i in coreml_nn.neuralNetworkClassifier.int64ClassLabels.vector)
        map_constructor_builder.add_attribute(attr_name, attr_content)
        nodes.append(map_constructor_builder.make_node())

    return nodes


def _resolve_name_conflicts(context, input, output):
    # CoreML sometime may use the same name for input/output. To resolve, we look for conflicts and rename the output.
    if isinstance(input, list) and isinstance(output, list):
        conflicts = [out for out in output if out in input]
        if len(conflicts) > 0:
            for i, out in enumerate(output):
                if out in conflicts:
                    output[i] = context.get_unique_name(out)
                    context.set_onnx_name(out, output[i])
    elif isinstance(input, str) and isinstance(output, str):
        if output == input:
            out = context.get_unique_name(output)
            context.set_onnx_name(output, out)
            output = out

    return output


def _convert_neural_network(context, coreml_nn):
    nodes = []
    inputs = coreml_nn.description.input

    # Determine the neural network type
    which_type = coreml_nn.WhichOneof('Type')
    nn_node = None
    if which_type == 'neuralNetworkRegressor':
        nn_node = coreml_nn.neuralNetworkRegressor
    elif which_type == 'neuralNetworkClassifier':
        nn_node = coreml_nn.neuralNetworkClassifier
        default_proba_tensor_name = None
        reserved_onnx_label_name = context.get_onnx_name(coreml_nn.description.predictedFeatureName)
        reserved_onnx_proba_name = context.get_onnx_name(coreml_nn.description.predictedProbabilitiesName)
    elif which_type == 'neuralNetwork':
        nn_node = coreml_nn.neuralNetwork

    # push the predicted feature name onto name map
    context.get_onnx_name(coreml_nn.description.predictedFeatureName)
    if which_type == 'neuralNetworkClassifier' and coreml_nn.description.predictedProbabilitiesName != '':
        # push the probability feature name onto name map
        context.get_onnx_name(coreml_nn.description.predictedProbabilitiesName)

    # Set the context data to be the CoreML node to allow for the preprocessors and layers to access
    context.data = coreml_nn

    for nn_pp in nn_node.preprocessing:
        # This generates onnx nodes for each preprocessor. To properly
        # connect the preprocessor to other preprocessors or to other
        # layers, we are using an input-output map where the input name
        # maps to the last output name
        input = nn_pp.featureName
        if not input:
            # If the featureName is empty, we will assume the input is the first feature
            input = inputs[0].name

        input = context.get_onnx_name(input)
        # Set output to None so that the output name will be generated
        output = context.get_unique_name(input)
        converter = registration.get_nn_converter(
            nn_pp.WhichOneof('preprocessor'))
        converted_pp = _do_convert(context, converter, nn_pp, input, output)

        # Update the latest output
        context.set_onnx_name(input, output)
        nodes.extend(converted_pp)

    for nn_layer in nn_node.layers:
        # iterate through the inputs of the layer looking for an input that
        # maps into the input_output map. If one is found, update the input with the
        # latest output of the map.
        inputs = [context.get_onnx_name(nn_input)
                  for nn_input in nn_layer.input]
        outputs = []
        for nn_output in nn_layer.output:
            new_name = context.get_unique_name(nn_output)
            context._onnx_map[nn_output] = new_name
            outputs.append(new_name)
        default_proba_tensor_name = outputs[0]
        converter = registration.get_nn_converter(nn_layer.WhichOneof('layer'))
        converted_layer = _do_convert(
            context, converter, nn_layer, inputs, outputs)
        nodes.extend(converted_layer)

    # CoreML uses different mechanism to calculate predicted class label and classes' probabilities than ONNX,
    # so we add some post-processing nodes for simulate their behavior.
    if which_type == 'neuralNetworkClassifier':
        nodes.extend(_create_post_processing_nodes(context, coreml_nn, default_proba_tensor_name, reserved_onnx_label_name, reserved_onnx_proba_name))
        context._onnx_map[coreml_nn.description.predictedFeatureName] = reserved_onnx_label_name
        context._onnx_map[coreml_nn.description.predictedProbabilitiesName] = reserved_onnx_proba_name

    return nodes


def _convert_pipeline(context, cm_node):
    nodes = []
    which_type = cm_node.WhichOneof('Type')
    if which_type == 'pipelineClassifier':
        pipeline_models = cm_node.pipelineClassifier.pipeline.models
    elif which_type == 'pipelineRegressor':
        pipeline_models = cm_node.pipelineRegressor.pipeline.models
    elif which_type == 'pipeline':
        pipeline_models = cm_node.pipeline.models
    else:
        raise RuntimeError('Unsupported CoreML pipeline type: {0}'.format(which_type))

    for cm_node in pipeline_models:
        converted_nodes = _convert_coreml_node(context, cm_node)
        nodes.extend(converted_nodes)

    return nodes


def _convert_coreml_node(context, cm_node):
    try:
        node_type = cm_node.WhichOneof('Type')
    except AttributeError:
        node_type = type(cm_node)
    except ValueError:
        node_type = type(cm_node)

    if node_type in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        return _convert_pipeline(context, cm_node)
    elif node_type in ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork']:
        return _convert_neural_network(context, cm_node)
    else:
        inputs = [context.get_onnx_name(input.name)
                  for input in cm_node.description.input]
        outputs = []
        for output in cm_node.description.output:
            new_name = context.get_unique_name(output.name)
            context.set_onnx_name(output.name, new_name)
            outputs.append(new_name)

        converter = registration.get_converter(node_type)
        return _do_convert(context, converter, cm_node, inputs, outputs)
