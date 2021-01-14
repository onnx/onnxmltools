# SPDX-License-Identifier: Apache-2.0

from ....common._registration import register_converter


def convert_mean_variance_normalization(scope, operator, container):
    op_type = 'MeanVarianceNormalization'
    op_name = scope.get_unique_operator_name(op_type)
    params = operator.raw_operator.mvn
    attrs = {'name': op_name}
    attrs['across_channels'] = params.acrossChannels
    attrs['normalize_variance'] = params.normalizeVariance
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('mvn', convert_mean_variance_normalization)
