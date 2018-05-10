# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .....proto import onnx_proto
from ....common._apply_operation import apply_batch_norm, apply_instance_norm
from ....common._registration import register_converter


def convert_batch_normalization(scope, operator, container):
    params = operator.raw_operator.batchnorm

    if params.instanceNormalization and not params.computeMeanVar:
        raise ValueError('It is impossible to do instance normalization without re-computing mean and variance')

    if params.instanceNormalization and params.computeMeanVar:
        op_type = 'InstanceNormalization'
    else:
        op_type = 'BatchNormalization'

    inputs = [operator.inputs[0].full_name]
    outputs = [operator.outputs[0].full_name]
    scale_tensor_name = scope.get_unique_variable_name(op_type + '_scale')
    container.add_initializer(scale_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                              params.gamma.floatValue)
    inputs.append(scale_tensor_name)
    bias_tensor_name = scope.get_unique_variable_name(op_type + '_B')
    container.add_initializer(bias_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels], params.beta.floatValue)
    inputs.append(bias_tensor_name)

    epsilon = params.epsilon

    if op_type == 'BatchNormalization':
        mean_tensor_name = scope.get_unique_variable_name(op_type + '_mean')
        container.add_initializer(mean_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                                  params.mean.floatValue)
        inputs.append(mean_tensor_name)
        variance_tensor_name = scope.get_unique_variable_name(op_type + '_variance')
        container.add_initializer(variance_tensor_name, onnx_proto.TensorProto.FLOAT, [params.channels],
                                  params.variance.floatValue)
        inputs.append(variance_tensor_name)
        momentum = 0.
        spatial = 1  # True

        if not params.instanceNormalization and params.computeMeanVar:
            # In this case, we apply batch normalization and adjust the statistics stored according the the batch
            # being processed.

            # To update "mean" and "var," we put their updated results back to the associated input tensors.
            outputs += inputs[1:3]
            # We also allocate two extra output buffers to store some intermediate results, but they are not used
            # in CoreML model.
            outputs.append(scope.get_unique_variable_name('saved_mean'))
            outputs.append(scope.get_unique_variable_name('saved_var'))
            # We choose "training" mode because some variables need to be updated.
            is_test = 0  # False
        elif not params.instanceNormalization and not params.computeMeanVar:
            # In this case, batch normalization is applied without updating mean, variance, etc. according to
            # the batches being processed. It means this operator works under testing model. Because there is no
            # variable update, we don't need to specify extra inputs and outputs like in previous code block.
            is_test = 1  # True
        else:
            raise ValueError('Unsupported operation mode')

        apply_batch_norm(scope, inputs, outputs, container, operator_name=operator.full_name, epsilon=epsilon,
                         is_test=is_test, momentum=momentum, spatial=spatial)
    else:
        apply_instance_norm(scope, inputs, outputs, container, operator_name=operator.full_name, epsilon=epsilon)


register_converter('batchnorm', convert_batch_normalization)
