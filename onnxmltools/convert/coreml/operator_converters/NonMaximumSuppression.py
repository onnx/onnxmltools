# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter


def convert_non_max_suppression(scope, operator, container):
    op_type = 'NonMaxSuppression'
    attrs = {'name': operator.full_name}

    raw_model = operator.raw_operator.nonMaxSuppression

    if raw_model.HasField('iouThreshold'):
        attrs['iou_threshold'] = float(raw_model.iouThreshold)
    if raw_model.HasField('confidenceThreshold'):
        attrs['score_threshold'] = float(raw_model.confidenceThreshold)

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name],
                       op_version=10, **attrs)


register_converter('nonMaxSuppression', convert_non_max_suppression)
