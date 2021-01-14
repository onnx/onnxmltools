# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter


def convert_lrn(scope, operator, container):
    op_type = 'LRN'
    params = operator.raw_operator.lrn
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs['size'] = params.localSize
    attrs['alpha'] = float(params.alpha)
    attrs['beta'] = float(params.beta)
    attrs['bias'] = float(params.k)
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('lrn', convert_lrn)
