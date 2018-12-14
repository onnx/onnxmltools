# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter


def convert_sklearn_concat(scope, operator, container):
    container.add_node('Concat', [s for s in operator.input_full_names],
                       operator.outputs[0].full_name, name=scope.get_unique_operator_name('Concat'), axis=1)


register_converter('SklearnConcat', convert_sklearn_concat)
