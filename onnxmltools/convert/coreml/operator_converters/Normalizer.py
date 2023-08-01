# SPDX-License-Identifier: Apache-2.0

from ...common._registration import register_converter


def convert_normalizer(scope, operator, container):
    op_type = 'Normalizer'
    attrs = {'name': operator.full_name}
    norms = ['MAX', 'L1', 'L2']
    norm_type = operator.raw_operator.normalizer.normType
    if norm_type in range(3):
        attrs['norm'] = norms[norm_type]
    else:
        raise ValueError('Invalid norm type: ' + norm_type)

    container.add_node(op_type, operator.input_full_names, operator.output_full_names, op_domain='ai.onnx.ml', **attrs)


register_converter('normalizer', convert_normalizer)
