# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import warnings
from ...proto import onnx
from ..common._container import CoremlModelContainer
from ..common._topology import Topology
from ..common.data_types import *


def _parse_coreml_feature(feature_info, target_opset, batch_size=1):
    '''
    Encode type information from CoreML's FeatureType protobuf message in converter's type system.

    Scalar types such as Int64FeatureType, DoubleFeatureType, and StringFeatureType in CoreML are interpreted as
    [batch_size, 1]-tensor. Tensor-like types such as ArrayFeature in CoreML is viewed as tensors with a prepend
    batch_size; for example, we use [batch_size, C, H, W] to denote [C, H, W]-array in CoreML.
    :param feature_info: CoreML FeatureDescription (https://apple.github.io/coremltools/coremlspecification/sections/DataStructuresAndFeatureTypes.html#featuretype)
    :param target_opset: the target ospet number in the converted model.
    :param batch_size: default batch size prepend to scalars and tensors variables from CoreML
    :return: one of our Int64Type, FloatType, StringType, Int64TensorType, FloatTensorType, or DictionaryType
    '''
    raw_type = feature_info.type
    doc_string = feature_info.shortDescription
    type_name = raw_type.WhichOneof('Type')

    if type_name == 'int64Type':
        return Int64Type(doc_string=doc_string)
    elif type_name == 'doubleType':
        return FloatType(doc_string=doc_string)
    elif type_name == 'stringType':
        return StringType(doc_string=doc_string)
    elif type_name == 'imageType':
        # Produce [C, H, W]-tensor, where C is the number of color channels, H the height, and W the width.
        color_space = raw_type.imageType.colorSpace
        shape = [batch_size]

        if doc_string:
            if doc_string[-1] not in ['.', '!', '?']:
                doc_string += '. '
            else:
                doc_string += ' '

        if color_space == 10:  # gray scale
            shape.append(1)
            doc_string += 'Image(s) in gray scale. If there are N images, it is a 4-D tensor with shape [N, 1, H, W]'
        elif color_space == 20:  # RGB (20)
            shape.append(3)
            doc_string += 'Image(s) in RGB format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the ' \
                          'C-axis are red, green, and blue channels, respectively.'
        elif color_space == 30:  # BGR (30)
            shape.append(3)
            doc_string += 'Image(s) in BGR format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the ' \
                          'C-axis are blue, green, and red channels, respectively.'
        else:
            raise ValueError('Unknown image format. Only gray-level, RGB, and BGR are supported')
        shape.append(raw_type.imageType.height)
        shape.append(raw_type.imageType.width)
        color_space_map = {10: 'Gray8', 20: 'Rgb8', 30: 'Bgr8'}
        return FloatTensorType(shape, color_space_map[color_space], doc_string=doc_string,
                               denotation='IMAGE', channel_denotations=['DATA_BATCH', 'DATA_CHANNEL', 'DATA_FEATURE', 'DATA_FEATURE'])
    elif type_name == 'multiArrayType':
        element_type_id = raw_type.multiArrayType.dataType
        shape = [d for d in raw_type.multiArrayType.shape]
        if len(shape) == 1:
            # [C]
            shape = [batch_size, shape[0]]
        elif len(shape) == 3:
            # [C, H, W]
            shape = [batch_size, shape[0], shape[1], shape[2]]
        else:
            shape = [batch_size, 1]  # Missing shape information. We will try inferring it.

        if element_type_id in [65568, 65600]:
            # CoreML FLOAT32 & DOUBLE
            return FloatTensorType(shape, doc_string=doc_string)
        elif element_type_id == 131104:
            # CoreML INT32
            return Int64TensorType(shape, doc_string=doc_string)
        else:
            raise ValueError('Invalid element type')
    elif type_name == 'dictionaryType':
        key_type = raw_type.dictionaryType.WhichOneof('KeyType')
        if key_type == 'int64KeyType':
            if target_opset < 7:
                return DictionaryType(Int64TensorType([1]), FloatTensorType([1]), doc_string=doc_string)
            else:
                return DictionaryType(Int64TensorType([]), FloatTensorType([]), doc_string=doc_string)
        elif key_type == 'stringKeyType':
            if target_opset < 7:
                return DictionaryType(StringTensorType([1]), FloatTensorType([1]), doc_string=doc_string)
            else:
                return DictionaryType(StringTensorType([]), FloatTensorType([]), doc_string=doc_string)
        else:
            raise ValueError('Unsupported key type: {}'.format(key_type))
    else:
        raise ValueError('Unsupported feature type: {}'.format(type_name))


def _parse_model(topology, scope, model, inputs=None, outputs=None):
    '''
    This is a delegate function of all top-level parsing functions. It does nothing but call a proper function
    to parse the given model.
    '''

    if inputs is None:
        inputs = list()
    if outputs is None:
        outputs = list()

    model_type = model.WhichOneof('Type')
    if model_type in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        _parse_pipeline_model(topology, scope, model, inputs, outputs)
    elif model_type in ['neuralNetworkClassifier', 'neuralNetworkRegressor', 'neuralNetwork']:
        _parse_neural_network_model(topology, scope, model, inputs, outputs)
    else:
        _parse_simple_model(topology, scope, model, inputs, outputs)


def _parse_simple_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a model containing only one operator (aka simple model).
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Create operator and then feed the model's inputs and outputs to the operator
        3. Connect local variables and their corresponding parent variables
    Note:
        1. Notice that a CoreML operator can contain no input and output, so we directly use model's inputs (outputs).
        2. Input and output names can be identical in CoreML, but they must be different for ONNX.
    '''

    # Create local scope for the considered model
    scope = topology.declare_scope('single', [parent_scope] + parent_scope.parent_scopes)

    # Create operator for the considered model
    this_operator = scope.declare_local_operator(model.WhichOneof('Type'), model)

    # Allocate inputs for the operator and then connect them with inputs from outside
    for var in model.description.input:
        # We assume that no duplicated raw name exists. Note that we set prepend=True because model inputs should
        # not hide any intermediate variables.
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size),
            prepend=True)
        this_operator.inputs.append(variable)

    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the first one to parent input.
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Allocate outputs for the operator and then connect them with outputs from outside
    for var in model.description.output:
        # We assume that no duplicated output raw name exists.
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size))
        this_operator.outputs.append(variable)

    # Connect local variables and variables passed into this scope. Our assumptions are described below.
    # 1. Assume a variable with 'A' as its CoreML name is passed in. There must be at least one local variable gets a
    #    raw name 'A'. That is, for each parent variable, at least one local duplicate is available.
    # 2. It's possible to find multiple local variables associated with the same raw name. For example, raw name 'A' can
    #    be associated with 'A' and 'A1' in ONNX. In this case, we connect the last one to parent output.
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_pipeline_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a pipeline including multiple sub-models.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the sub-models and create their inputs and outputs variables
        3. Connect model's (not sub-model's) inputs and outputs with proper variables created when parsing sub-models
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML sub-model can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
    '''

    # Create local scope
    scope = topology.declare_scope('pipeline', [parent_scope] + parent_scope.parent_scopes)

    # Use the same name to denote sub-models
    pipeline_type = model.WhichOneof('Type')
    if pipeline_type == 'pipelineClassifier':
        sub_models = model.pipelineClassifier.pipeline.models
    elif pipeline_type == 'pipelineRegressor':
        sub_models = model.pipelineRegressor.pipeline.models
    elif pipeline_type == 'pipeline':
        sub_models = model.pipeline.models
    else:
        raise ValueError('Unsupported CoreML pipeline type: {0}'.format(pipeline_type))

    # Sequentially parse the sub-models
    for sub_model in sub_models:
        # Declare the sub-model's input and output in this scope. Those input and output variables will be passed into
        # the sub-model's parsing function and connected with proper child variables.
        sub_inputs = []
        for var in sub_model.description.input:
            variable = scope.get_local_variable_or_declare_one(
                var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size))
            sub_inputs.append(variable)
        sub_outputs = []
        for var in sub_model.description.output:
            variable = scope.declare_local_variable(
                var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size))
            sub_outputs.append(variable)
        _parse_model(topology, scope, sub_model, sub_inputs, sub_outputs)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.input:
        # Find the first variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]
        # Create model's input variable. Note that we set prepend=True because model inputs should not hide any
        # intermediate variables.
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size),
            prepend=True)
        # Feed the input to the sub-model's input. It's possible to add type conversion here by using a casting operator
        # rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    # Declare the model's (not sub-model's) inputs and then link them with sub-model's inputs
    for var in model.description.output:
        # Find the latest variable with the same raw name declared when parsing the sub-models
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]
        # Create model's output variable
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size))
        # Connect the input and a sub-model's input. It's possible to add type conversion here by using a casting
        # operator rather than identity, but we haven't see the need of doing so in practices.
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def _parse_neural_network_model(topology, parent_scope, model, inputs, outputs):
    '''
    Parse a neural network model.
    Steps:
        1. Create local scope for allocating local variables and operators
        2. Sequentially parse the preprocessors and layers
        3. Connect model's (neither layers' nor preprocessors') inputs and outputs with proper variables created when
           parsing sub-models.
        4. Link local variables and the corresponding parent variables (only model's inputs and outputs are considered)
    Note:
        1. A CoreML preprocessor/layer can use the same variable for its input and output.
        2. Two CoreML variables may have the same name but different types.
        3. Preprocessor sometime may not include any information about its input
    '''

    # Create local scope to which all subsequent variables and operators belongs
    scope = topology.declare_scope('NeuralNetwork', [parent_scope] + parent_scope.parent_scopes)

    network = None
    network_type = model.WhichOneof('Type')
    if network_type == 'neuralNetworkClassifier':
        network = model.neuralNetworkClassifier
    elif network_type == 'neuralNetworkRegressor':
        network = model.neuralNetworkRegressor
    elif network_type == 'neuralNetwork':
        network = model.neuralNetwork
    else:
        raise ValueError('Unknown network type {}'.format(network_type))

    for op in network.preprocessing:
        operator = scope.declare_local_operator(op.WhichOneof('preprocessor') + 'Preprocessor', op)

        # Infer the variable name to be processed if feature name is an empty string
        name = op.featureName if op.featureName != '' else model.description.input[0].name

        # Find out input variable
        original = scope.get_local_variable_or_declare_one(name)
        original.type = FloatTensorType()  # A newly-declared variable has no type, so we add it.
        operator.inputs.append(original)

        # Declare a variable for storing the processed result
        processed = scope.declare_local_variable(name)
        processed.type = FloatTensorType()  # A newly-declared variable has no type, so we add it
        operator.outputs.append(processed)

    for op in network.layers:
        operator = scope.declare_local_operator(op.WhichOneof('layer'), op)

        # Find out input variable and connect them with the operator
        for name in op.input:
            variable = scope.get_local_variable_or_declare_one(name)
            # Although most neural network operators only accepts floats, we still need to handle the only exception,
            # embedding layer. In the furture, we should create a Cast operator right inside embedding's converter.
            if operator.type == 'embedding':
                variable.type = Int64TensorType()
            else:
                variable.type = FloatTensorType()
            operator.inputs.append(variable)

        # Declare variables for catching the operator's outputs
        for name in op.output:
            variable = scope.declare_local_variable(name)
            variable.type = FloatTensorType()  # A newly-declared variable has no type, so we add it
            operator.outputs.append(variable)

    sink_variables = scope.find_sink_variables()

    # Declare the model's inputs and outputs. Then, connect them with proper variables computed by the main network
    for var in model.description.input:
        # Search for the first variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][0]]

        # Declare model input. To prevent intermediate variables form being hidden by model inputs, prepend is True.
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size),
            prepend=True)

        # A heuristic which forces the input of embedding to be integer tensor rather than float tensor.
        # Ideally this should be done by adding a cast operator, but ONNX doesn't have float-to-int casting.
        # If this variable is produced by another component in a CoreML pipeline, a bug may occur especially
        # when the source component's output type is float tensor.
        if isinstance(child_variable.type, Int64TensorType):
            variable.type = Int64TensorType(variable.type.shape)

        # Feed model input to the associated model input
        operator_type = find_type_conversion(source_type=variable.type, target_type=child_variable.type)
        operator = scope.declare_local_operator(operator_type)
        operator.inputs.append(variable)
        operator.outputs.append(child_variable)

    # Connect local input variables with proper variables from parent scope
    for parent_variable in inputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][0]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(parent_variable)
        operator.outputs.append(child_variable)

    for var in model.description.output:
        # CoreML's predicted label is not connected with any operator, so we handle it later as a special case.
        special_variable_names = [model.description.predictedFeatureName, model.description.predictedProbabilitiesName]
        if model.WhichOneof('Type') == 'neuralNetworkClassifier' and var.name in special_variable_names:
            continue
        # Search for the latest variable (declared when parsing network layers) associated with the considered raw name
        child_variable = scope.variables[scope.variable_name_mapping[var.name][-1]]

        # Create model output variable
        variable = scope.declare_local_variable(
            var.name, _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size))

        # Feed result calculated by the network to the output variable
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(variable)

    # If predicted label exists, connect probability tensor and label by a special operator
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedFeatureName:
        # Find out the description of predicted label and declare a label variable
        label_variable = None
        for var in model.description.output:
            if var.name == model.description.predictedFeatureName:
                label_type = _parse_coreml_feature(var, topology.target_opset, topology.default_batch_size)
                label_variable = scope.declare_local_variable(var.name, label_type)
                break
        operator = scope.declare_local_operator('tensorToLabel', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])
        operator.outputs.append(label_variable)

    # Probability tensor is implicitly converted into a dictionary (i.e., map) in CoreML. We handle this case here.
    if model.WhichOneof('Type') == 'neuralNetworkClassifier' and model.description.predictedProbabilitiesName:
        operator = scope.declare_local_operator('tensorToProbabilityMap', model)

        probability_name = model.description.predictedProbabilitiesName
        if probability_name in scope.variable_name_mapping:
            # Find the latest probability variable
            operator.inputs.append(scope.variables[scope.variable_name_mapping[probability_name][-1]])
        else:
            # If predicted probability tensor is missing in CoreML model, it defaults to the first sink of the network
            operator.inputs.append(sink_variables[0])

        # Find out the description of predicted probabilities and declare a variable for probability map
        for var in model.description.output:
            if var.name == model.description.predictedProbabilitiesName:
                probability_type = _parse_coreml_feature(var, topology.target_opset,
                                                         topology.default_batch_size)
                probability_variable = scope.declare_local_variable(var.name, probability_type)
                operator.outputs.append(probability_variable)
                break

    # Connect local output variables with proper variables from parent scope
    for parent_variable in outputs:
        raw_name = parent_variable.raw_name
        child_variable = scope.variables[scope.variable_name_mapping[raw_name][-1]]
        operator = scope.declare_local_operator('identity')
        operator.inputs.append(child_variable)
        operator.outputs.append(parent_variable)


def parse_coreml(model, initial_types=None, target_opset=None, custom_conversion_functions=None, custom_shape_calculators=None):
    '''
    This is the root function of the whole parsing procedure.
    :param model: CoreML model
    :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
    name and a type defined in data_types.py.
    :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: a Topology object. It's a intermediate representation of the input CoreML model
    '''

    # Add model-level input and output names into a set. The set will be fed into our Topology so that all its elements
    # will not be used to declare variables
    reserved_variable_names = set()
    for var in list(model.description.input) + list(model.description.output):
        reserved_variable_names.add(var.name)

    # Determine the batch size for parsing CoreML model's input and output features. Note that batch size is always
    # missing in all CoreML models.
    default_batch_size = 'None'

    # Topology is shared by both of CoreML and scikit-learn conversion frameworks, so we have a wrapper class,
    # CoremlModelContainer, to make sure our topology-related functions can seamlessly handle both of CoreML and
    # scikit-learn.
    topology = Topology(CoremlModelContainer(model),
                        default_batch_size,
                        initial_types,
                        reserved_variable_names,
                        target_opset=target_opset,
                        custom_conversion_functions=custom_conversion_functions,
                        custom_shape_calculators=custom_shape_calculators)
    scope = topology.declare_scope('__root__')

    # Instead of using CoremlModelContainer, we directly pass the model in because _parse_model is CoreML-specific.
    _parse_model(topology, scope, model)
    topology.compile()

    for variable in topology.find_root_and_sink_variables():
        color_space = getattr(variable.type, 'color_space', None)
        if color_space:
            if topology.metadata_props.setdefault('Image.BitmapPixelFormat', color_space) != color_space:
                warnings.warn('Conflicting pixel formats found. In ONNX, all input/output images must use the same pixel format.')
        # Use original CoreML names for model-level input(s)/output(s)
        if variable.raw_name not in reserved_variable_names:
            continue
        topology.rename_variable(variable.onnx_name, variable.raw_name)
    return topology
