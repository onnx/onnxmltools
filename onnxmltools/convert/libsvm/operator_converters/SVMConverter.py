#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#--------------------------------------------------------------------------

from ....proto import onnx_proto
from ...common._registration import register_converter
from ...common.utils import cast_list

import svm
import svmutil
import numpy


class SVMConverter:
    """
    Converts a SVM model trained with *svmlib*.
    """
    @staticmethod
    def validate(svm_node):
        try:
            hasattr(svm_node, 'param')
            hasattr(svm_node, 'SV')
            hasattr(svm_node, 'nSV')
            hasattr(svm_node, 'sv_coef')
            hasattr(svm_node, 'l')
            hasattr(svm_node.param, 'gamma')
            hasattr(svm_node.param, 'coef0')
            hasattr(svm_node.param, 'degree')
            hasattr(svm_node.param, 'kernel_type')
            hasattr(svm_node, 'rho')
        except AttributeError as e:
            raise RuntimeError("Missing type from svm node:" + str(e))


    @staticmethod
    def get_sv(svm_node):
        labels = svm_node.get_labels()
        sv = svm_node.get_SV()
        if len(sv) == 0:
            raise RuntimeError("No support vector machine. This usually happens with very small datasets or the training failed.")

        maxk = max(max(row.keys() for row in sv))
        mat = numpy.zeros((len(sv), maxk+1), dtype=numpy.float32)

        for i, row in enumerate(sv):
            for k,v in row.items():
                if k == -1:
                    k = 0
                try:
                    mat[i, k] = v
                except IndexError:
                    raise RuntimeError("Issue with one dimension\nlabels={0}\n#sv={1}\nshape={2}\npos={3}x{4}-maxk={5}-svm.l={6}\nrow={7}".format(labels, nsv, mat.shape, i, k, maxk, svm_node.l, row))
        # We do not consider the first row (class -1).
        mat = mat[:, 1:]

        # mat.shape should be (n_vectors, X.shape[1])
        # where X.shape[1] is the number of features.
        # However, it can be <= X.shape.[1] if the last
        # every coefficient is null on the last column.
        # To fix that, an extra parameter must be added to
        # the convert function as there is no way to guess
        # that information from svmlib model.
        return numpy.array(mat.ravel(), dtype=float)

    @staticmethod
    def convert(operator, scope, container, svm_node, inputs, model_name, nb_class):
        kt = svm_node.param.kernel_type
        if kt == svm.RBF:
            kt = 'RBF'
        elif kt == svm.SIGMOID:
            kt = 'SIGMOID'
        elif kt == svm.POLY:
            kt = 'POLY'
        elif kt == svm.LINEAR:
            kt = "LINEAR"
        else:
            raise RuntimeError("Unexpected value for kernel: {0}".format(kt))

        def copy_sv_coef(sv_coef):
            nrc = svm_node.nr_class-1
            res = numpy.zeros((svm_node.l, nrc), dtype=numpy.float64)
            for i in range(0, svm_node.l):
                for j in range(nrc):
                    res[i, j] = svm_node.sv_coef[j][i]
            return res.T

        if nb_class > 2:
            # See above.
            coef = copy_sv_coef(svm_node.sv_coef)
        else:
            coef = numpy.array(svm_node.get_sv_coef()).ravel()

        atts = dict(kernel_type=kt,
                    kernel_params=[float(_) for _ in [svm_node.param.gamma, svm_node.param.coef0, svm_node.param.degree]],
                    coefficients=list(coef.ravel()))

        return dict(node='SVMConverter', inputs=operator.input_full_names,
                    outputs = [o.full_name for o in operator.outputs],
                    op_domain='ai.onnx.ml', attrs=atts)

class SVCConverter(SVMConverter):

    @staticmethod
    def validate(svm_node):
        SVMConverter.validate(svm_node)
        try:
            hasattr(svm_node, 'probA')
            hasattr(svm_node, 'probB')
        except AttributeError as e:
            raise RuntimeError("Missing type from svm node:" + str(e))

    @staticmethod
    def convert(operator, scope, container, svm_node, inputs):
        nbclass = len(svm_node.get_labels())
        # See converter for sklearn.
        nb = SVMConverter.convert(operator, scope, container, svm_node, inputs, "SVMClassifier", nbclass)
        sign_rho = -1.
        st = svm_node.param.svm_type

        if svm_node.is_probability_model():
            if st == svm.C_SVC or st == svm.NU_SVC:
                n_class = len(svm_node.get_labels())
                n = int(n_class*(n_class-1)/2)
                probA = [svm_node.probA[i] for i in range(n)]
                probB = [svm_node.probB[i] for i in range(n)]
                nb["attrs"]["prob_a"] = probA
                nb["attrs"]["prob_b"] = probB
                nb["attrs"]['rho'] = [svm_node.rho[i] * sign_rho for i in range(n)]
            else:
                nb["attrs"]['rho'] = [svm_node.rho[0] * sign_rho]
        elif st == svm.C_SVC or st == svm.NU_SVC:
            n_class = len(svm_node.get_labels())
            n = int(n_class*(n_class-1)/2)
            nb["attrs"]['rho'] = [svm_node.rho[i] * sign_rho for i in range(n)]
        else:
            nb["attrs"]['rho'] = [svm_node.rho[0] * sign_rho]

        class_labels = cast_list(int, svm_node.get_labels())
        # Predictions are different when label are not sorted (multi-classification).
        class_labels.sort()
        nb["attrs"]['classlabels_ints'] = class_labels
        output_type = onnx_proto.TensorProto.INT64

        if len(nb['outputs']) != 2:
            raise RuntimeError("The model outputs label and probabilities not {0}".format(nb['outputs']))

        nbclass = len(svm_node.get_labels())
        nb["attrs"]['vectors_per_class'] = [svm_node.nSV[i] for i in range(nbclass)]
        nb["attrs"]['post_transform'] = "NONE"
        nb["attrs"]['support_vectors'] = SVCConverter.get_sv(svm_node)

        # Add a vec dictionizer to handle the map output
        container.add_node('SVMClassifier', nb['inputs'],
                           nb['outputs'], op_domain='ai.onnx.ml',
                           name=scope.get_unique_operator_name('SVMClassifier'),
                           **nb['attrs'])


class SVRConverter(SVMConverter):

    @staticmethod
    def validate(svm_node):
        SVMConverter.validate(svm_node)
        try:
            hasattr(svm_node, 'l')
        except AttributeError as e:
            raise RuntimeError("Missing type from svm node:" + str(e))

    @staticmethod
    def convert(operator, scope, container, svm_node, inputs):
        nb = SVMConverter.convert(operator, scope, container, svm_node, inputs, "SVMRegressor", 0)

        nb['attrs']["n_supports"] =  svm_node.l
        nb['attrs']['post_transform'] = "NONE"
        nb['attrs']['rho'] = [-svm_node.rho[0]]
        nb['attrs']['support_vectors'] = SVCConverter.get_sv(svm_node)

        container.add_node('SVMRegressor', nb['inputs'],
                           nb['outputs'], op_domain='ai.onnx.ml',
                           name=scope.get_unique_operator_name('SVMRegressor'),
                           **nb['attrs'])


class AnyLibSvmConverter:

    @staticmethod
    def select(svm_node):
        if svm_node.param.svm_type in (svm.C_SVC, svm.NU_SVC):
            return SVCConverter
        if svm_node.param.svm_type in (svm.EPSILON_SVR, svm.NU_SVR):
            return SVRConverter
        raise RuntimeError("svm_node type is unexpected '{0}'".format(svm_node.param.svm_type))

    @staticmethod
    def validate(svm_node):
        sel = AnyLibSvmConverter.select(svm_node)
        sel.validate(svm_node)

    @staticmethod
    def convert(operator, scope, container, svm_node, inputs):
        sel = AnyLibSvmConverter.select(svm_node)
        sel.convert(operator, scope, container, svm_node, inputs)


def convert_libsvm(scope, operator, container):

    inputs = operator.inputs
    model = operator.raw_operator
    converter = AnyLibSvmConverter
    onnx_nodes = []
    outputs = None
    converter.validate(model)
    converter.convert(operator, scope, container, model, inputs)


# Register the class for processing
register_converter("LibSvmSVC", convert_libsvm)
register_converter("LibSvmSVR", convert_libsvm)
