"""
Tests scikit-linear converter.
"""
import unittest
import numpy
import onnxmltools
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from sklearn.svm import LinearSVC
from onnxmltools import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model


class TestSklearnSVM(unittest.TestCase):

    def _fit_model_binary_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X

    def _fit_binary_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X[:5].astype(numpy.float32)

    def _fit_model_multiclass_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model, X

    def _fit_multi_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        tx = numpy.vstack([X[:2], X[60:62], X[110:112], X[147:149]]).astype(numpy.float32)
        return model, tx

    def _fit_multi_classification2(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[-5:] = 3
        model.fit(X, y)
        tx = numpy.vstack([X[:2], X[60:62], X[110:112], X[147:149]]).astype(numpy.float32)
        return model, tx

    def _fit_multi_regression(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = numpy.vstack([iris.target, iris.target]).T
        model.fit(X, y)
        return model, X[:5].astype(numpy.float32)

    def _check_attributes(self, node, attribute_test):
        attributes = node.attribute
        attribute_map = {}
        for attribute in attributes:
            attribute_map[attribute.name] = attribute

        for k, v in attribute_test.items():
            self.assertTrue(k in attribute_map)
            if v is not None:
                attrib = attribute_map[k]
                if isinstance(v, str):
                    self.assertEqual(attrib.s, v.encode(encoding='UTF-8'))
                elif isinstance(v, int):
                    self.assertEqual(attrib.i, v)
                elif isinstance(v, float):
                    self.assertEqual(attrib.f, v)
                elif isinstance(v, list):
                    self.assertEqual(attrib.ints, v)
                else:
                    self.fail('Unknown type')

    def test_convert_svmc_linear_binary(self):
        model, X = self._fit_binary_classification(SVC(kernel='linear', probability=False))
        model_onnx = convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'LINEAR',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnBinSVCLinearPF-NoProb-Opp",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.4')")

    def test_convert_svmr_linear_binary(self):
        model, X = self._fit_binary_classification(SVR(kernel='linear'))
        model_onnx = convert_sklearn(model, 'SVR', [('input', FloatTensorType([1, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        self._check_attributes(nodes[0], {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'LINEAR',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegSVRLinear-Dec3")

    def test_convert_nusvmc_binary(self):
        model, X = self._fit_binary_classification(NuSVC(probability=False))
        model_onnx = convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'RBF',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnBinNuSVCPF-NoProb-Opp",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.4')")

    def test_convert_nusvmr_binary(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(model, 'SVR', [('input', FloatTensorType([1, X.shape[1]]))])
        node = model_onnx.graph.node[0]
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'RBF',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR")

    def test_registration_convert_nusvr_model(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = onnxmltools.convert_sklearn(model, 'SVR', [('input', FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR2")

    def test_registration_convert_nusvc_model_multi(self):
        model, X = self._fit_multi_classification(NuSVC(probability=True))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclNuSVCPT", verbose=True)

    def test_registration_convert_svc_model(self):
        model, X = self._fit_binary_classification(SVC(kernel='linear', probability=True))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnBinNuSVCPT")

    def test_model_linear_svc_binary_class(self):
        model, X = self._fit_model_binary_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'linear SVC', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearSVCBinary-NoProb")

    def test_model_linear_svc_multi_class(self):
        model, X = self._fit_model_multiclass_classification(LinearSVC())
        model_onnx = convert_sklearn(model, 'multi-class linear SVC', [('input', FloatTensorType([1, 3]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx, basename="SklearnLinearSVCMulti")

    @unittest.skip(reason="scikit-learn applies _ovr_decision_function. It does not follow ONNX spec.")
    def test_convert_svmc_multi(self):
        model, X = self._fit_multi_classification2(SVC(probability=False))
        model_onnx = convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'RBF',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclNuSVCPF")

    @unittest.skip(reason="scikit-learn applies _ovr_decision_function. It does not follow ONNX spec.")
    def test_convert_svmc_linear_multi(self):
        model, X = self._fit_multi_classification2(SVC(kernel='linear', probability=False))
        model_onnx = convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'LINEAR',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclSVCLinearPF")

    @unittest.skip(reason="scikit-learn applies _ovr_decision_function. It does not follow ONNX spec.")
    def test_registration_convert_svc_model_multi(self):
        model, X = self._fit_multi_classification2(SVC(probability=False))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVC', [('input', FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnMclSVCPFMulti")


if __name__ == "__main__":
    # TestSklearnSVM().test_model_linear_svc_binary_class()
    unittest.main()
