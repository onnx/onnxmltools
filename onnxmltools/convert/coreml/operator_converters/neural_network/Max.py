# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_max
from ....common._registration import register_converter


def convert_max(scope, operator, container):
    apply_max(scope, operator.input_full_names, operator.output_full_names,
              container, operator_name=operator.full_name)


register_converter('max', convert_max)
