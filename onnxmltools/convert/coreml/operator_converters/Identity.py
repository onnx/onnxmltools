# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter


def convert_identity(scope, operator, container):
    container.add_node('Identity', operator.input_full_names, operator.output_full_names, name=operator.full_name)


register_converter('identity', convert_identity)
