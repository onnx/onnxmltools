#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sys
from uuid import uuid4
from ..common import model_util
from ...proto import onnx_proto
from ... import __domain__
from ... import __producer__
from ... import __producer_version__
from ... import __model_version__


class ModelBuilder:
    def __init__(self, name=None, doc_string=''):
        self._name = str(uuid4().hex) if name is None else name
        self._doc_string = doc_string
        self._inputs = []
        self._outputs = []
        self._nodes = []
        self._initializers = []
        self._values = []
        self._op_sets = set()

    def add_inputs(self, inputs):
        self._inputs.extend(inputs)

    def add_outputs(self, outputs):
        self._outputs.extend(outputs)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def add_initializers(self, initializers):
        self._initializers.extend(initializers)

    def add_values(self, values):
        self._values.extend(values)

    def add_op_set(self, op_set):
        self._op_sets.add(op_set)

    def make_model(self):
        return model_util.make_model(self._name,
                                     onnx_proto.IR_VERSION,
                                     __producer__,
                                     __producer_version__,
                                     __domain__,
                                     __model_version__,
                                     self._doc_string,
                                     self._op_sets,
                                     self._nodes,
                                     self._inputs,
                                     self._outputs,
                                     self._values,
                                     self._initializers)
