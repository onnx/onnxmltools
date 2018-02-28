#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from sklearn import linear_model
from sklearn import svm
from .common import add_zipmap
from .common import add_normalizer
from ..common import register_converter
from ..common import NodeBuilder
from ..common import utils
from ..common import model_util
from ...proto import onnx_proto
import numpy as np


class GLMClassifierConverter:
    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'coef_')
            utils._check_has_attr(sk_node, 'intercept_')
            utils._check_has_attr(sk_node, 'classes_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        if isinstance(sk_node.coef_, np.ndarray):
            coefficients = sk_node.coef_.flatten().tolist()
        intercepts = sk_node.intercept_.tolist()
        classes = sk_node.classes_
        if len(classes) == 2:
            coefficients = list(map(lambda x: -1 * x, coefficients)) + coefficients
            intercepts = list(map(lambda x: -1 * x, intercepts)) + intercepts

        multi_class = 0
        if hasattr(sk_node, 'multi_class'):
            if sk_node.multi_class == 'ovr':
                multi_class = 1
            else:
                multi_class = 2

        nb = NodeBuilder(context, 'LinearClassifier', op_domain='ai.onnx.ml')
        nb.add_attribute('coefficients', coefficients)
        nb.add_attribute('intercepts', intercepts)
        nb.add_attribute('multi_class', multi_class == 2)
        if sk_node.__class__.__name__ == 'LinearSVC':
            nb.add_attribute('post_transform', 'NONE')
        else:
            if multi_class == 2:
                nb.add_attribute('post_transform', 'SOFTMAX')
            else:
                nb.add_attribute('post_transform', 'LOGISTIC')

        if utils.is_string_type(classes):
            class_labels = utils.cast_list(str, classes)
            nb.add_attribute('classlabels_strings', class_labels)
            output_type = onnx_proto.TensorProto.STRING
        else:
            class_labels = utils.cast_list(int, classes)
            nb.add_attribute('classlabels_ints', class_labels)
            output_type = onnx_proto.TensorProto.INT64

        nb.extend_inputs(inputs)
        output_y = model_util.make_tensor_value_info(nb.name, output_type, [1, 1])
        nb.add_output(output_y)
        context.add_output(output_y)

        prob_input = context.get_unique_name('classProbability')
        nb.add_output(prob_input)

        output_name = prob_input
        appended_node_normalizer = None

        # Add normalizer in the case of multi-class.
        if multi_class > 0 and sk_node.__class__.__name__ != 'LinearSVC':
            appended_node_normalizer, output_name = add_normalizer(prob_input, output_type, "L1", context)

        # Add a ZipMap to handle the map output.
        if len(classes) > 2 or sk_node.__class__.__name__ != 'LinearSVC':
            appended_node_zipmap = add_zipmap(output_name, output_type, class_labels, context)
        else:
            score_selector = NodeBuilder(context, 'Slice', op_version=2)
            score_selector.add_input(output_name)
            select_output = context.get_unique_name(output_name)
            score_selector.add_output(select_output)
            score_selector.add_attribute('starts', [0, 1])
            score_selector.add_attribute('ends', [1, 2])
            selector_output = model_util.make_tensor_value_info(select_output, onnx_proto.TensorProto.FLOAT, [1])
            context.add_output(selector_output)
            appended_node_zipmap = score_selector.make_node()

        if appended_node_normalizer != None:
            return [nb.make_node(), appended_node_normalizer, appended_node_zipmap]
        else:
            return [nb.make_node(), appended_node_zipmap]


# Register the class for processing
register_converter(svm.LinearSVC, GLMClassifierConverter)
register_converter(linear_model.LogisticRegression, GLMClassifierConverter)
register_converter(linear_model.SGDClassifier, GLMClassifierConverter)
