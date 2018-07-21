# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, InputLayer
from ...proto import onnx
from ..common._container import KerasModelContainer
from ..common._topology import Topology
from ..common.data_types import *


def _extract_inbound_nodes(model):
    if hasattr(model, 'inbound_nodes'):
        return model.inbound_nodes
    elif hasattr(model, '_inbound_nodes'):
        return model._inbound_nodes
    else:
        raise ValueError('Failed to find inbound_nodes and _inbound_nodes when parsing Keras model')


def extract_model_input_and_output_shapes(model, default_batch_size):
    if hasattr(model, 'input_shape'):
        if not isinstance(model.input_shape, list):
            input_shapes = [list(model.input_shape)]
        else:
            input_shapes = [list(shape) for shape in model.input_shape]
    elif hasattr(model, 'input_shapes'):
        input_shapes = [list(shape) for shape in model.input_shapes]
    else:
        raise ValueError('Fail to extract model input shape(s)')

    for shape in input_shapes:
        if not isinstance(shape[0], numbers.Integral):
            shape[0] = default_batch_size

    if hasattr(model, 'output_shape'):
        if not isinstance(model.output_shape, list):
            output_shapes = [list(model.output_shape)]
        else:
            output_shapes = [list(shape) for shape in model.output_shape]
    elif hasattr(model, 'output_shapes'):
        output_shapes = [list(shape) for shape in model.output_shapes]
    else:
        raise ValueError('Fail to extract model output shape(s)')

    for shape in output_shapes:
        if not isinstance(shape[0], numbers.Integral):
            shape[0] = default_batch_size

    return input_shapes, output_shapes


def determine_tensor_type(tensor, default_batch_size, keras_shape=None):
    # keras_shape can overwrite the shaped defined in Tensorflow tensor
    if keras_shape is None:
        tensor_shape = [d.value if d.value is not None else 'None' for d in tensor.shape]
    else:
        tensor_shape = [d if d is not None else 'None' for d in keras_shape]

    # Adjust batch size if needed
    if tensor_shape[0] == 'None':
        tensor_shape[0] = default_batch_size

    # Determine the tensor's element type
    tensor_type = tensor.dtype
    if tensor_type in [tf.int8, tf.int16, tf.int32, tf.int64]:
        return Int64TensorType(shape=tensor_shape)
    elif tensor_type in [tf.float16, tf.float32, tf.float64]:
        return FloatTensorType(shape=tensor_shape)
    else:
        raise ValueError('Unable to find out a correct type for tensor %s' % tensor)


def parse_keras(model, initial_types=None, targeted_onnx=onnx.__version__, custom_conversion_functions=None, custom_shape_calculators=None):
    '''
    The main parsing function of Keras Model and Sequential objects.

    :param model: A Keras Model or Sequential object
    :param initial_types: A list providing some types for some root variables. Each element is a tuple of a variable
    name and a type defined in data_types.py.
    :param targeted_onnx: a version string such as `1.1.2` or `1.2.1` for specifying the ONNX version used to produce
    the output model.
    :param custom_conversion_functions: a dictionary for specifying the user customized conversion function
    :param custom_shape_calculators: a dictionary for specifying the user customized shape calculator
    :return: a Topology object. It's a intermediate representation of the input Keras model
    '''
    raw_model_container = KerasModelContainer(model)

    topology = Topology(raw_model_container, default_batch_size=1, initial_types=initial_types, targeted_onnx=targeted_onnx,
                        custom_conversion_functions=custom_conversion_functions, custom_shape_calculators=custom_shape_calculators)
    scope = topology.declare_scope('__root__')

    # Each inbound node defines an evaluation of the underlining model (if the model is called multiple times, it may
    # contain several inbound nodes). According to the tensors specified in those inbound nodes, we declare the roots
    # and leaves of the computational graph described by the Keras input model.
    for node in _extract_inbound_nodes(model):
        input_shapes, output_shapes = extract_model_input_and_output_shapes(model, topology.default_batch_size)

        # Declare inputs for a specific model execution
        for tensor, shape in zip(node.input_tensors, input_shapes):
            raw_model_container.add_input_name(tensor.name)
            tensor_type = determine_tensor_type(tensor, topology.default_batch_size, list(shape))
            scope.get_local_variable_or_declare_one(tensor.name, tensor_type)

        # Declare outputs for a specific model execution
        for tensor, shape in zip(node.output_tensors, output_shapes):
            raw_model_container.add_output_name(tensor.name)
            tensor_type = determine_tensor_type(tensor, topology.default_batch_size, list(shape))
            scope.get_local_variable_or_declare_one(tensor.name, tensor_type)

    # For each model execution, we call a parsing function to create a computational (sub-)graph because ONNX has no
    # model/layer sharing.
    for node in _extract_inbound_nodes(model):
        _parse_keras(topology, scope, model, node)

    topology.root_names = [variable.onnx_name for variable in scope.variables.values()]

    return topology


def _parse_keras(topology, parent_scope, model, inbound_node):
    if isinstance(model, Model):
        scope = topology.declare_scope('scope')
        # Declare output variables so that they can be connected with the variables produced in layers and sub-models
        for layer in model.layers:
            for node in _extract_inbound_nodes(layer):
                for tensor in node.output_tensors:
                    tensor_type = determine_tensor_type(tensor, topology.default_batch_size)
                    scope.declare_local_variable(tensor.name, tensor_type)

        # Recursively call the parsing function
        for layer in model.layers:
            for node in _extract_inbound_nodes(layer):
                _parse_keras(topology, scope, layer, node)

        # Connect the variables declared when parsing the input model and the actual model inputs. inbound_node has the
        # actual inputs while the while graph is indeed declared only via the first inbound node of the input model.
        # That is, for a shared (sub-)model, we may declare it several times and each time we may connect its I/O with
        # the I/O specified in a inbound node.
        for parent_tensor, local_tensor in zip(inbound_node.input_tensors, _extract_inbound_nodes(model)[0].input_tensors):
            parent_tensor_type = determine_tensor_type(parent_tensor, topology.default_batch_size)
            local_tensor_type = determine_tensor_type(local_tensor, topology.default_batch_size)
            parent_variable = parent_scope.get_local_variable_or_declare_one(parent_tensor.name, parent_tensor_type)
            local_variable = scope.get_local_variable_or_declare_one(local_tensor.name, local_tensor_type)
            operator = scope.declare_local_operator('identity')
            operator.inputs.append(parent_variable)
            operator.outputs.append(local_variable)

        # Connect the variables declared when parsing the input model and the actual model outputs. inbound_node has the
        # actual outputs while the while graph is indeed declared via the first inbound node of the input models.
        for parent_tensor, local_tensor in zip(inbound_node.output_tensors, _extract_inbound_nodes(model)[0].output_tensors):
            parent_tensor_type = determine_tensor_type(parent_tensor, topology.default_batch_size)
            local_tensor_type = determine_tensor_type(local_tensor, topology.default_batch_size)
            parent_variable = parent_scope.get_local_variable_or_declare_one(parent_tensor.name, parent_tensor_type)
            local_variable = scope.get_local_variable_or_declare_one(local_tensor.name, local_tensor_type)
            operator = scope.declare_local_operator('identity')
            operator.inputs.append(local_variable)
            operator.outputs.append(parent_variable)

    elif isinstance(model, Layer):
        if isinstance(model, InputLayer):
            return
        operator = parent_scope.declare_local_operator(type(model), raw_model=model)

        # Simply connect the layer's I/O with variables declared in the parent scope. Note that it may create input
        # variables in the parent scope because we only declare output variables in the beginning of _parse_keras(...)
        for parent_tensor in inbound_node.input_tensors:
            tensor_type = determine_tensor_type(parent_tensor, topology.default_batch_size)
            operator.inputs.append(parent_scope.get_local_variable_or_declare_one(parent_tensor.name, tensor_type))
        for parent_tensor in inbound_node.output_tensors:
            tensor_type = determine_tensor_type(parent_tensor, topology.default_batch_size)
            operator.outputs.append(parent_scope.get_local_variable_or_declare_one(parent_tensor.name, tensor_type))

    else:
        raise RuntimeError('Unsupported Keras component %s' % type(model))
