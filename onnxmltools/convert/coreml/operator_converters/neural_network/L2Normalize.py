# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ....common._registration import register_converter


def convert_l2_normalization(scope, operator, container):
    # The first dimension is batch size, so the normalization is done along the 2nd axis (indexed by 1).
    attrs = {'name': operator.full_name, 'axis': 1, 'p': 2}  # Caffe normalization happens per image in one batch
    container.add_node('LpNormalization', operator.input_full_names, operator.output_full_names, **attrs)


register_converter('l2normalize', convert_l2_normalization)
