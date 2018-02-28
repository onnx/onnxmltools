#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

class Node:
    def __init__(self,
                 onnx_node,
                 inputs,
                 outputs,
                 initializers,
                 values,
                 op_domain,
                 op_version):

        self._onnx_node = onnx_node
        self._inputs = inputs
        self._outputs = outputs
        self._initializers = initializers
        self._values = values
        self._domain = op_domain
        self._version = op_version

    @property
    def name(self):
        return self._onnx_node.name

    @property
    def attributes(self):
        return self._onnx_node.attribute

    @property
    def input_names(self):
        return self._onnx_node.input

    @input_names.setter
    def input_names(self, value):
        self._onnx_node.input = value

    @property
    def output_names(self):
        return self._onnx_node.output

    @output_names.setter
    def output_names(self, value):
        self._onnx_node.output = value

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def initializers(self):
        return self._initializers

    @property
    def values(self):
        return self._values

    @property
    def onnx_node(self):
        return self._onnx_node

    @property
    def domain_version_pair(self):
        return (self._domain, self._version)

