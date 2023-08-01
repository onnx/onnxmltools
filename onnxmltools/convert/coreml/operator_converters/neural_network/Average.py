# SPDX-License-Identifier: Apache-2.0

from ....common._apply_operation import apply_mean
from ....common._registration import register_converter


def convert_average(scope, operator, container):
    apply_mean(scope, operator.input_full_names, operator.output_full_names, container,
               operator_name=operator.full_name)


register_converter('average', convert_average)
