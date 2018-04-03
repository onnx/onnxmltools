# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from uuid import uuid4
from ...proto import onnx_proto
from ..common import utils
from ..common._container import RawModelContainer
from ..common._topology import convert_topology
from ._parse import parse_keras

# Register conversion functions and shape inference functions
from . import operator_converters
from . import shape_calculators

class KerasModelContainer(RawModelContainer):

    def __init__(self, keras_model):
        super(KerasModelContainer, self).__init__(keras_model)
        self._input_raw_names = set()
        self._output_raw_names = set()

    def add_input_name(self, name):
        self._input_raw_names.add(name)

    def add_output_name(self, name):
        self._output_raw_names.add(name)

    @property
    def input_names(self):
        return [name for name in self._input_raw_names]

    @property
    def output_names(self):
        return [name for name in self._output_raw_names]


def convert(model, name=None, doc_string=''):
    topology = parse_keras(model)

    topology.compile()

    if name is None:
        name = str(uuid4().hex)

    onnx_model = convert_topology(topology, name, doc_string)

    return onnx_model

