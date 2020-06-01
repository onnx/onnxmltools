# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxconverter_common.container import (
    RawModelContainer,
    CommonSklearnModelContainer
)


class LightGbmModelContainer(CommonSklearnModelContainer):
    pass


class XGBoostModelContainer(CommonSklearnModelContainer):
    pass


class H2OModelContainer(CommonSklearnModelContainer):
    pass


class SparkmlModelContainer(RawModelContainer):

    def __init__(self, sparkml_model):
        super(SparkmlModelContainer, self).__init__(sparkml_model)
        # Sparkml models have no input and output specified, so we create them and store them in this container.
        self._inputs = []
        self._outputs = []

    @property
    def input_names(self):
        return [variable.raw_name for variable in self._inputs]

    @property
    def output_names(self):
        return [variable.raw_name for variable in self._outputs]

    def add_input(self, variable):
        # The order of adding variables matters. The final model's input names are sequentially added as this list
        if variable not in self._inputs:
            self._inputs.append(variable)

    def add_output(self, variable):
        # The order of adding variables matters. The final model's output names are sequentially added as this list
        if variable not in self._outputs:
            self._outputs.append(variable)


class CoremlModelContainer(RawModelContainer):

    def __init__(self, coreml_model):
        super(CoremlModelContainer, self).__init__(coreml_model)

    @property
    def input_names(self):
        return [str(var.name) for var in self.raw_model.description.input]

    @property
    def output_names(self):
        return [str(var.name) for var in self.raw_model.description.output]


class LibSvmModelContainer(CommonSklearnModelContainer):
    pass
