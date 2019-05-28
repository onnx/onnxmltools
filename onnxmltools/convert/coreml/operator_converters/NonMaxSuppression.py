# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter
import numpy as np

def convert_non_max_suppression(scope, operator, container):
    op_type = 'NonMaxSuppression'
    attrs = {'name': operator.full_name}

    raw_model = operator.raw_operator.nonMaxSuppression

    if raw_model.HasField('coordinatesInputFeatureName'):
        attrs['boxes'] = np.array(raw_model.coordinatesInputFeatureName).astype(np.float32)
    if raw_model.HasField('confidenceInputFeatureName'):
        attrs['scores'] = np.array(raw_model.confidenceInputFeatureName).astype(np.float32)

    if raw_model.HasField('iouThreshold'):
        attrs['iou_threshold'] = np.array(raw_model.iouThreshold).astype(np.float32)
    if raw_model.HasField('confidenceThreshold'):
        attrs['score_threshold'] = np.array(raw_model.confidenceThreshold).astype(np.float32)

    if raw_model.HasField('PickTop') and raw_model.PickTop.HasField('perClass'):
        attrs['max_output_boxes_per_class'] = np.array(raw_model.PickTop.perClass).astype(np.float32)

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name],
                       op_version=10, **attrs)


register_converter('nonMaxSuppression', convert_non_max_suppression)
