# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ..common import register_converter
from ..common import utils
from ..common import NodeBuilder


class ImputerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'imputer')
        except AttributeError as e:
            raise RuntimeError("Missing type from CoreML node:" + str(e))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        """
        Converts a CoreML Imputer to ONNX
        """
        nb = NodeBuilder(context, 'Imputer')

        if cm_node.imputer.HasField('replaceDoubleValue'):
            nb.add_attribute('replaced_value_float',
                             cm_node.imputer.replaceDoubleValue)
        elif cm_node.imputer.HasField('replaceInt64Value'):
            nb.add_attribute('replaced_value_int64',
                             cm_node.imputer.replaceInt64Value)
        if cm_node.imputer.HasField('imputedDoubleArray'):
            nb.add_attribute('imputed_value_floats',
                             cm_node.imputer.imputedDoubleArray.vector)
        elif cm_node.imputer.HasField('imputedInt64Array'):
            nb.add_attribute('imputed_value_int64s',
                             cm_node.imputer.imputedInt64Array.vector)

        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        return nb.make_node()


# Register the class for processing
register_converter("imputer", ImputerConverter)
