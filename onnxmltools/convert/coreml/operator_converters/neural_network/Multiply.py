# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._apply_operation import apply_mul
from ....common._registration import register_converter


def convert_multiply(scope, operator, container):
    if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
        broadcast = 1
    else:
        broadcast = 0

    apply_mul(scope, operator.input_full_names, operator.output_full_names, container,
              operator_name=operator.full_name, broadcast=broadcast)


register_converter('multiply', convert_multiply)
