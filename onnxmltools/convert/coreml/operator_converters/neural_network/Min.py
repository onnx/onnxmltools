# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_min
from ....common._registration import register_converter


def convert_min(scope, operator, container):
    apply_min(
        scope,
        operator.input_full_names,
        operator.output_full_names,
        container,
        operator.full_name,
    )


register_converter("min", convert_min)
