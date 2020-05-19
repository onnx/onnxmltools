# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ...common._registration import register_converter
import numpy as np

def convert_non_max_suppression(scope, operator, container):
    if container.target_opset < 10:
        raise RuntimeError('NonMaxSuppression is only support in Opset 10 or higher')

    op_type = 'NonMaxSuppression'
    attrs = {'name': operator.full_name}

    raw_model = operator.raw_operator.nonMaximumSuppression

    # Attribute: center_point_box
    # The box data is supplied as [x_center, y_center, width, height], similar to TF models.
    attrs['center_point_box'] = 1

    # We index into scope.variable_name_mapping with key of raw_model.inputFeatureName to extract list of values.
    # Assuming there's only one NMS node in this graph, the last node will be the input we're looking for.
    # If there's more than one NMS coreml model, then return error.
    if raw_model.HasField('coordinatesInputFeatureName'):
        coordinates_input = scope.variable_name_mapping[raw_model.coordinatesInputFeatureName]
        if len(coordinates_input) > 1:
            raise RuntimeError('NMS conversion does not currently support more than one NMS node in an ONNX graph')
        attrs['boxes'] = np.array(coordinates_input[0]).astype(np.float32)
    if raw_model.HasField('confidenceInputFeatureName'):
        confidence_input = scope.variable_name_mapping[raw_model.confidenceInputFeatureName]
        if len(coordinates_input) > 1:
            raise RuntimeError('NMS conversion does not currently support more than one NMS node in an ONNX graph')
        attrs['scores'] = np.array(confidence_input[0]).astype(np.float32)

    if raw_model.HasField('iouThreshold'):
        attrs['iou_threshold'] = np.array(raw_model.iouThreshold).astype(np.float32)
    if raw_model.HasField('confidenceThreshold'):
        attrs['score_threshold'] = np.array(raw_model.confidenceThreshold).astype(np.float32)

    if raw_model.HasField('PickTop') and raw_model.PickTop.HasField('perClass'):
        attrs['max_output_boxes_per_class'] = np.array(raw_model.PickTop.perClass).astype(np.float32)

    container.add_node(op_type, [operator.inputs[0].full_name], [operator.outputs[0].full_name],
                       op_version=10, **attrs)


register_converter('nonMaxSuppression', convert_non_max_suppression)
