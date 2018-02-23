#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common import NodeBuilder
from ...common import utils
from ...common import model_util
from ...common import registration


class BatchnormLayerConverter:

    @staticmethod
    def validate(cm_node):
        try:
            utils._check_has_attr(cm_node, 'batchnorm')
            utils._check_has_attr(cm_node, 'input')
            utils._check_has_attr(cm_node, 'output')
        except AttributeError as e:
            raise RuntimeError('Missing attribute in neural network layer: {0}'.format(cm_node.name))

    @staticmethod
    def convert(context, cm_node, inputs, outputs):
        params = cm_node.batchnorm

        if params.instanceNormalization and not params.computeMeanVar:
            raise ValueError('It is impossible to do instance normalization without computing mean and variance')

        if params.instanceNormalization and params.computeMeanVar:
            op_type = 'InstanceNormalization'
        else:
            op_type = 'BatchNormalization'

        nb = NodeBuilder(context, op_type)
        nb.extend_inputs(inputs)
        nb.extend_outputs(outputs)

        dims = [params.channels]
        scale_tensor = model_util.make_tensor('scale', onnx_proto.TensorProto.FLOAT, dims, params.gamma.floatValue)
        nb.add_initializer(scale_tensor)

        bias_tensor = model_util.make_tensor('B', onnx_proto.TensorProto.FLOAT, dims, params.beta.floatValue)
        nb.add_initializer(bias_tensor)

        epsilon = 1e-5
        if params.epsilon != 0:
            epsilon = 1e-5
        nb.add_attribute('epsilon', epsilon)
        nb.add_attribute('spatial', True)

        if op_type == 'BatchNormalization':
            mean_tensor = model_util.make_tensor('mean', onnx_proto.TensorProto.FLOAT, dims, params.mean.floatValue)
            nb.add_initializer(mean_tensor)

            variance_tensor = model_util.make_tensor('var', onnx_proto.TensorProto.FLOAT, dims,
                                                     params.variance.floatValue)
            nb.add_initializer(variance_tensor)

            nb.add_attribute('momentum', 0.)

            if not params.instanceNormalization and params.computeMeanVar:
                # In this case, we apply batch normalization and adjust the statistics stored according the the batch
                # being processed.

                # To update "mean" and "var," we put their updated results back to the associated input tensors.
                extra_outputs = [nb.input_names[1:3]]

                # We also allocate two extra output buffers to store some intermediate results, but they are not used
                # in CoreML model.
                extra_outputs.append(context.get_unique_name('saved_mean'))
                extra_outputs.append(context.get_unique_name('saved_var'))
                nb.extend_outputs(extra_outputs)

                # We set "training" mode because some variables need to be updated.
                nb.add_attribute('is_test', False)
            elif not params.instanceNormalization and not params.computeMeanVar:
                # In this case, batch normalization is applied without updating mean, variance, etc. according to
                # the batches being processed. It means this operator works under testing model. Because there is no
                # variable update, we don't need to specify extra inputs and outputs like in previous code block.
                nb.add_attribute('is_test', True)
        else:
            nb.add_attribute('is_test', True)

        return nb.make_node()


registration.register_nn_converter('batchnorm', BatchnormLayerConverter)
