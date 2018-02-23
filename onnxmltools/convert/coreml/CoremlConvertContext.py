#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import ConvertContext

class CoremlConvertContext(ConvertContext):
    '''
    The ConvertContext provides data about the conversion, specifically keeping a mapping of the old->new names as
    well as provides helper functions for generating unique names
    '''

    def __init__(self, parent = None):
        ConvertContext.__init__(self)
        self._onnx_map = {}
        self._parent = parent
        self._data = None
        if parent is None:
            self._top_level_inputs = []
            self._top_level_outputs = []
        else:
            self._top_level_inputs = parent._top_level_inputs
            self._top_level_outputs = parent._top_level_outputs

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def top_level_inputs(self):
        return self._top_level_inputs

    @top_level_inputs.setter
    def top_level_inputs(self, inputs):
        self._top_level_inputs = inputs

    @property
    def top_level_outputs(self):
        return self._top_level_outputs

    @top_level_outputs.setter
    def top_level_outputs(self, outputs):
        self._top_level_outputs = outputs

    def extend_top_level_inputs(self, inputs):
        self._top_level_inputs.extend(inputs)

    def extend_top_level_outputs(self, outputs):
        self._top_level_outputs.extend(outputs)

    def clear_data(self):
        self._data = None

    def get_unique_name(self, name):
        if self._parent is not None:
            return self._parent.get_unique_name(name)
        return ConvertContext.get_unique_name(self, name)

    def set_onnx_name(self, old, new):
        self._onnx_map[old] = new

    def get_onnx_name(self, name, default_name=None):
        onnx_name = None
        if name in self._onnx_map:
            onnx_name = self._onnx_map[name]
        elif self._parent is not None:
            onnx_name = self._parent.get_onnx_name(name, default_name)

        if onnx_name is not None:
            return onnx_name

        # generate the name
        gen_name = default_name
        if gen_name is None:
            gen_name = name
        onnx_name = self.get_unique_name(gen_name)
        self.set_onnx_name(name, onnx_name)
        return onnx_name

