#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sys
from uuid import uuid4
from ..common import model_util
from ...proto import onnx_proto

class ModelBuilder:
    def __init__(self, name=None, doc_string='', metadata_props=[]):
        self._name = str(uuid4().hex) if name is None else name
        self._doc_string = doc_string
        self._metadata_props = metadata_props
        self._inputs = []
        self._outputs = []
        self._nodes = []
        self._initializers = []
        self._values = []
        self._operator_domain_version_pairs = set()

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

    def add_domain_version_pair(self, pair):
        self._operator_domain_version_pairs.add(pair)

    def make_model(self):
        return model_util.make_model(self._name,
                                     onnx_proto.IR_VERSION,
                                     model_util.producer(),
                                     model_util.producer_version(),
                                     model_util.domain(),
                                     model_util.model_version(),
                                     self._doc_string,
                                     self._metadata_props,
                                     self._operator_domain_version_pairs,
                                     self._nodes,
                                     self._inputs,
                                     self._outputs,
                                     self._values,
                                     self._initializers)
