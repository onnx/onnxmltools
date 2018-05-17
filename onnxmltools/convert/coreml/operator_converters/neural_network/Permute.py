# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._apply_operation import apply_transpose
from ....common._registration import register_converter


def convert_permute(scope, operator, container):
    apply_transpose(scope, operator.input_full_names, operator.output_full_names, container,
                    operator_name=operator.full_name, perm=operator.raw_operator.permute.axis)


register_converter('permute', convert_permute)
