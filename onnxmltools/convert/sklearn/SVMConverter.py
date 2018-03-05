#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

import sklearn.svm
import numpy as np
from .common import add_zipmap
from ...proto import onnx_proto
from ..common import register_converter
from ..common import utils
from ..common import NodeBuilder
from ..common import model_util


class SVMConverter:
    """
    Converts a SVM model.
    See `SVR <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_,
    `SVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.   
    """

    @staticmethod
    def validate(sk_node):
        try:
            utils._check_has_attr(sk_node, 'support_vectors_')
            utils._check_has_attr(sk_node, 'coef0')
            utils._check_has_attr(sk_node, '_gamma')
            utils._check_has_attr(sk_node, 'degree')
            utils._check_has_attr(sk_node, 'intercept_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs, model_name, classes=None):
        coef = sk_node.dual_coef_.ravel().tolist()
        support_vectors = sk_node.support_vectors_.ravel().tolist()
        intercept = sk_node.intercept_
        if classes == 2:
            coef = np.negative(coef)
            intercept = np.negative(intercept)
        attrs = dict(kernel_type=sk_node.kernel.upper(),
                     kernel_params=[float(_) for _ in [sk_node._gamma, sk_node.coef0, sk_node.degree]],
                     support_vectors=support_vectors, coefficients=coef,
                     rho=intercept)

        nb = NodeBuilder(context, model_name, op_domain='ai.onnx.ml')
        for k, v in attrs.items():
            nb.add_attribute(k, v)

        nb.extend_inputs(inputs)
        return nb


class SVCConverter(SVMConverter):
    @staticmethod
    def validate(sk_node):
        SVMConverter.validate(sk_node)
        try:
            utils._check_has_attr(sk_node, 'classes_')
            utils._check_has_attr(sk_node, 'n_support_')
            utils._check_has_attr(sk_node, 'probA_')
            utils._check_has_attr(sk_node, 'probB_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        classes = sk_node.classes_
        nb = SVMConverter.convert(context, sk_node, inputs, "SVMClassifier", len(classes))
        if len(sk_node.probA_) > 0:
            nb.add_attribute("prob_a", sk_node.probA_)
        if len(sk_node.probB_) > 0:
            nb.add_attribute("prob_b", sk_node.probB_)

        nb.add_attribute('vectors_per_class', sk_node.n_support_)

        if utils.is_numeric_type(classes):
            class_labels = utils.cast_list(int, classes)
            nb.add_attribute('classlabels_ints', class_labels)
            output_type = onnx_proto.TensorProto.INT64
        elif utils.is_string_type(classes):
            class_labels = utils.cast_list(str, classes)
            nb.add_attribute('classlabels_strings', class_labels)
            output_type = onnx_proto.TensorProto.STRING
        else:
            raise RuntimeError("Invalid class type:" + classes.dtype)

        nb.add_attribute('post_transform', 'NONE')

        output_y = model_util.make_tensor_value_info(nb.name, output_type, [1, 1])
        nb.add_output(output_y)
        context.add_output(output_y)

        # Add a ZipMap to handle the map output
        prob_input = context.get_unique_name('classProbability')
        nb.add_output(prob_input)
        appended_node = add_zipmap(prob_input, output_type, class_labels, context)
        return [nb.make_node(), appended_node]


class SVRConverter(SVMConverter):

    @staticmethod
    def validate(sk_node):
        SVMConverter.validate(sk_node)
        try:
            utils._check_has_attr(sk_node, 'support_')
        except AttributeError as e:
            raise RuntimeError("Missing type from sklearn node:" + str(e))

    @staticmethod
    def convert(context, sk_node, inputs):
        nb = SVMConverter.convert(context, sk_node, inputs, "SVMRegressor")
        nb.add_attribute('post_transform', "NONE")
        output_dim = None
        try:
            if len(inputs[0].type.tensor_type.shape.dim) > 0:
                output_dim = [1, inputs[0].type.tensor_type.shape.dim[0].dim_value]
        except AttributeError as e:
            raise ValueError('Invalid or missing input dimension.')
        nb.add_attribute('n_supports', len(sk_node.support_))
        nb.add_output(model_util.make_tensor_value_info(nb.name, onnx_proto.TensorProto.FLOAT, output_dim))
        return nb.make_node()


# Register the class for processing
register_converter(sklearn.svm.SVC, SVCConverter)
register_converter(sklearn.svm.SVR, SVRConverter)
register_converter(sklearn.svm.NuSVC, SVCConverter)
register_converter(sklearn.svm.NuSVR, SVRConverter)
