# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_concat
from ....common._registration import register_converter


def convert_concat(scope, operator, container):
    if operator.raw_operator.concat.sequenceConcat:
        axis = 0
    else:
        axis = 1

    apply_concat(
        scope,
        operator.input_full_names,
        operator.output_full_names,
        container,
        operator_name=operator.full_name,
        axis=axis,
    )


register_converter("concat", convert_concat)
