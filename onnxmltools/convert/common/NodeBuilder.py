# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from . import utils
from ...proto import onnx_proto
from .Node import Node


class NodeBuilder:
    def __init__(self, context, op_type, op_domain='', op_version=1):
        self._op_type = op_type
        self._name = context.get_unique_name(op_type)
        self._attributes = {}
        self._input_names = []
        self._inputs = []
        self._output_names = []
        self._outputs = []
        self._initializers = []
        self._values = []
        self._context = context
        self._op_domain = op_domain
        self._op_version = op_version

    @property
    def name(self):
        return self._name

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    def add_input(self, input):
        if utils.is_string_type(input):
            self._input_names.append(input)
        elif isinstance(input, onnx_proto.ValueInfoProto):
            self._input_names.append(input.name)
            self._inputs.append(input)
        else:
            raise Exception('Unsupported input type')

    def add_empty_input(self):
        self.add_input('')

    def add_output(self, output):
        if utils.is_string_type(output):
            self._output_names.append(output)
        elif isinstance(output, onnx_proto.ValueInfoProto):
            self._output_names.append(output.name)
            self._outputs.append(output)
        else:
            raise Exception('Unsupported output type')

    def extend_inputs(self, inputs):
        for input in inputs:
            self.add_input(input)

    def extend_outputs(self, outputs):
        for output in outputs:
            self.add_output(output)

    def add_attribute(self, name, value):
        self._attributes[name] = value

    def add_initializer(self, tensor, tensor_name=None):
        '''
        Set the name of the initializer to be node-name.tensor-name if tensor_name is not specified. Otherwise, using
        the tensor_name as the initializer's name.
        '''
        if tensor_name is None:
            tensor.name = self._name + '.' + tensor.name
        else:
            tensor.name = tensor_name
        self._initializers.append(tensor)
        self._input_names.append(tensor.name)

    def add_value(self, value):
        '''
        Set the name of the initializer to be node-name.value-name
        '''
        value_name = self._name + '.' + value.name
        value.name = value_name
        self._input_names.append(value_name)
        self._values.append(value)

    def make_node(self):
        from . import model_util
        # Create a ONNX node based on the information we have
        onnx_node = model_util.make_node(self._op_type,
                                         self._input_names,
                                         self._output_names,
                                         self._name,
                                         self._op_domain,
                                         **self._attributes)
        # Add the operator set of this operator
        op_set = onnx_proto.OperatorSetIdProto()
        op_set.domain = self._op_domain
        op_set.version = self._op_version
        # Pass a high-level node upon the ONNX node we just created
        node = Node(onnx_node,
                    self._inputs,
                    self._outputs,
                    self._initializers,
                    self._values,
                    op_set)
        return node
