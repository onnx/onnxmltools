# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


from ....common._registration import register_converter


def convert_preprocessing_mean_image(scope, operator, container):
    op_type = 'MeanSubtraction'
    op_name = scope.get_unique_operator_name(op_type)
    attrs = {'name': op_name, 'image': operator.raw_operator.meanImage}
    container.add_node(op_type, operator.input_full_names, operator.output_full_names, **attrs)


register_converter('meanImagePreprocessor', convert_preprocessing_mean_image)
