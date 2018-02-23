"""
Tests scikit-linear converter.
"""
import unittest
import onnxmltools
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from onnxmltools.convert.sklearn.SVMConverter import SVCConverter, SVRConverter
from onnxmltools.convert.sklearn.SklearnConvertContext import SklearnConvertContext as ConvertContext
from onnxmltools.proto import onnx_proto
from onnxmltools.convert.common import model_util


class TestSklearnSVM(unittest.TestCase):
    defaultInput = [model_util.make_tensor_value_info('input', onnx_proto.TensorProto.FLOAT, [1])]

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
        attributes = node.attributes
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

    def _check_outputs(self, node, output_names):
        outputs = node.outputs
        output_map = {}
        for output in outputs:
            output_map[output.name] = output

        for name in output_names:
            self.assertTrue(name in output_map)

    def test_convert_svmc_linear_binary(self):
        model = self._fit_binary_classification(SVC(kernel='linear', probability=False))
        context = ConvertContext()
        nodes = SVCConverter.convert(context, model, self.defaultInput)
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
        self._check_outputs(svc_node, [svc_node.name])

    def test_convert_svmc_linear_multi(self):
        model = self._fit_multi_classification(SVC(kernel='linear', probability=False))
        context = ConvertContext()
        nodes = SVCConverter.convert(context, model, self.defaultInput)
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
        self._check_outputs(svc_node, [svc_node.name])

    def test_convert_svmr_linear_binary(self):
        model = self._fit_binary_classification(SVR(kernel='linear'))
        context = ConvertContext()
        node = SVRConverter.convert(context, model, self.defaultInput)
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'LINEAR',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})
        self._check_outputs(node, [node.name])

    def test_convert_svmr_linear_multi(self):
        model = self._fit_multi_classification(SVR(kernel='linear'))
        context = ConvertContext()
        node = SVRConverter.convert(context, model, self.defaultInput)
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'LINEAR',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})
        self._check_outputs(node, [node.name])

    def test_convert_nusvmc_binary(self):
        model = self._fit_binary_classification(NuSVC(probability=False))
        context = ConvertContext()
        nodes = SVCConverter.convert(context, model, self.defaultInput)
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
        self._check_outputs(svc_node, [svc_node.name])

    def test_convert_nusvmc_multi(self):
        model = self._fit_multi_classification(NuSVC(probability=False))
        context = ConvertContext()
        nodes = SVCConverter.convert(context, model, self.defaultInput)
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
        self._check_outputs(svc_node, [svc_node.name])

    def test_convert_nusvmr_binary(self):
        model = self._fit_binary_classification(NuSVR())
        context = ConvertContext()
        node = SVRConverter.convert(context, model, self.defaultInput)
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'RBF',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_convert_nusvmr_multi(self):
        model = self._fit_multi_classification(NuSVR())
        context = ConvertContext()
        node = SVRConverter.convert(context, model, self.defaultInput)
        self.assertIsNotNone(node)
        self._check_attributes(node, {'coefficients': None,
                                      'kernel_params': None,
                                      'kernel_type': 'RBF',
                                      'post_transform': None,
                                      'rho': None,
                                      'support_vectors': None})

    def test_registration_convert_nusvr_model(self):
        model = self._fit_binary_classification(NuSVR())
        model_onnx = onnxmltools.convert_sklearn(model)
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_nusvc_model(self):
        model = self._fit_multi_classification(NuSVC(probability=False))
        model_onnx = onnxmltools.convert_sklearn(model)
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_svr_model(self):
        model = self._fit_multi_classification(SVR(kernel='linear'))
        model_onnx = onnxmltools.convert_sklearn(model)
        self.assertIsNotNone(model_onnx)

    def test_registration_convert_svc_model(self):
        model = self._fit_binary_classification(SVC(kernel='linear', probability=False))
        model_onnx = onnxmltools.convert_sklearn(model)
        self.assertIsNotNone(model_onnx)
