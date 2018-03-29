"""
Tests scikit-linear converter.
"""
import unittest
import onnxmltools
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from onnxmltools import convert_sklearn
from onnxmltools.convert.common._data_types import FloatTensorType


class TestSklearnSVM(unittest.TestCase):

    def _fit_binary_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model

    def _fit_multi_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        model.fit(X, y)
        return model

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
        model = self._fit_binary_classification(SVC(kernel='linear', probability=False))
        nodes = convert_sklearn(model, 'SVC', [FloatTensorType([1, 1])]).graph.node
        self.assertIsNotNone(nodes)
        self.assertEqual(len(nodes), 2)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'LINEAR',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})

    def test_convert_svmc_linear_multi(self):
        model = self._fit_multi_classification(SVC(kernel='linear', probability=False))
        nodes = convert_sklearn(model, 'SVC', [FloatTensorType([1, 1])]).graph.node
        self.assertIsNotNone(nodes)
        self.assertEqual(len(nodes), 2)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'LINEAR',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})

    def test_convert_svmr_linear_binary(self):
        model = self._fit_binary_classification(SVR(kernel='linear'))
        nodes = convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])]).graph.node
        self.assertIsNotNone(nodes)
        self._check_attributes(nodes[0], {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'LINEAR',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_convert_svmr_linear_multi(self):
        model = self._fit_multi_classification(SVR(kernel='linear'))
        node = convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])]).graph.node[0]
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'LINEAR',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_convert_nusvmc_binary(self):
        model = self._fit_binary_classification(NuSVC(probability=False))
        nodes = convert_sklearn(model, 'SVC', [FloatTensorType([1, 1])]).graph.node
        self.assertIsNotNone(nodes)
        self.assertEqual(len(nodes), 2)

        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'RBF',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})

    def test_convert_nusvmc_multi(self):
        model = self._fit_multi_classification(NuSVC(probability=False))
        nodes = convert_sklearn(model, 'SVC', [FloatTensorType([1, 1])]).graph.node
        self.assertIsNotNone(nodes)
        self.assertEqual(len(nodes), 2)
        svc_node = nodes[0]
        self._check_attributes(svc_node, {'coefficients': None,
                                          'kernel_params': None,
                                          'kernel_type': 'RBF',
                                          'post_transform': None,
                                          'rho': None,
                                          'support_vectors': None,
                                          'vectors_per_class': None})

    def test_convert_nusvmr_binary(self):
        model = self._fit_binary_classification(NuSVR())
        node = convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])]).graph.node[0]
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'RBF',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_convert_nusvmr_multi(self):
        model = self._fit_multi_classification(NuSVR())
        node = convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])]).graph.node[0]
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'RBF',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_registration_convert_nusvr_model(self):
        model = self._fit_binary_classification(NuSVR())
        model_onnx = onnxmltools.convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])])
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_nusvc_model(self):
        model = self._fit_multi_classification(NuSVC(probability=False))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVC', [FloatTensorType([1, 1])])
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_svr_model(self):
        model = self._fit_multi_classification(SVR(kernel='linear'))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])])
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_svc_model(self):
        model = self._fit_binary_classification(SVC(kernel='linear', probability=False))
        model_onnx = onnxmltools.convert_sklearn(model, 'SVR', [FloatTensorType([1, 1])])
        self.assertIsNotNone(model_onnx)
