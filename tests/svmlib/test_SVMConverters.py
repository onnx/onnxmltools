"""
Tests scikit-linear converter.
"""
import tempfile
import numpy
import svm
import numpy as np
import unittest
from sklearn.datasets import load_iris
from onnxmltools.convert.libsvm import convert
from svm import C_SVC as SVC, EPSILON_SVR as SVR, NU_SVC as NuSVC, NU_SVR as NuSVR
import svmutil
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import dump_data_and_model

try:
    from svm import PRINT_STRING_FUN, print_null
    noprint = PRINT_STRING_FUN(print_null)
except ImportError:
    # This was recently added.
    noprint = None


class SkAPI:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X, options=""):
        if hasattr(X, 'shape'):
            X = X.tolist()
        return svmutil.svm_predict([0 for i in X], list(X), self.model, options=options)
        
    def __getstate__(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        svmutil.svm_save_model(f.name, self.model)
        with open(f.name, "rb") as h:
            return {'data': h.read()}
        os.remove(f)
        
    def __setstate__(self, data):
        f = tempfile.NamedTemporaryFile(delete=False)
        with open(f.name, "wb") as h:
            h.write(data['by'])
        self.model = svmutil.svm_load_model(f.name)
        os.remove(f)
        

class SkAPIReg(SkAPI):
    def predict(self, X):
        res = SkAPI.predict(self, X)
        ret = numpy.array(res[0]).ravel()
        return ret.reshape(X.shape[0], len(ret) // X.shape[0]).astype(numpy.float32)


class SkAPIClProba2(SkAPI):
    
    def predict(self, X):
        res = SkAPI.predict(self, X)
        ret = numpy.array(res[0]).ravel()
        return ret

    def predict_proba(self, X):
        res = SkAPI.predict(self, X, options="-b 1")
        pro = numpy.array(res[-1]).ravel()
        pro = pro.reshape(X.shape[0], len(pro) // X.shape[0]).astype(numpy.float32)
        return pro


class SkAPICl(SkAPI):
    
    def predict(self, X):
        res = SkAPI.predict(self, X)
        ret = numpy.array(res[0]).ravel()
        return ret

    def decision_function(self, X):
        res = SkAPI.predict(self, X)
        pro = numpy.array(res[-1]).ravel()
        pro = pro.reshape(X.shape[0], len(pro) // X.shape[0]).astype(numpy.float32)
        return pro


class TestSvmLibSVM(unittest.TestCase):

    def test_convert_svmc_linear(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = SVC
        param.kernel_type = svmutil.LINEAR
        param.eps = 1
        param.probability = 1
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmcLinear", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIClProba2(libsvm_model), node,
                            basename="LibSvmSvmcLinear-Dec3")

    def test_convert_svmc(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = SVC
        param.kernel_type = svmutil.RBF
        param.eps = 1
        param.probability = 1
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmc", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIClProba2(libsvm_model), node,
                            basename="LibSvmSvmc-Dec3")

    def test_convert_svmr_linear(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        prob = svmutil.svm_problem(y, X.tolist())        
                
        param = svmutil.svm_parameter()
        param.svm_type = SVR
        param.kernel_type = svmutil.LINEAR
        param.eps = 1
        if noprint:
            param.print_func = noprint

        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmrLinear", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIReg(libsvm_model), node,
                            basename="LibSvmSvmrLinear-Dec3")

    def test_convert_svmr(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        prob = svmutil.svm_problem(y, X.tolist())        
                
        param = svmutil.svm_parameter()
        param.svm_type = SVR
        param.kernel_type = svmutil.RBF
        param.probability = 1
        param.eps = 1
        if noprint:
            param.print_func = noprint

        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmr", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIReg(libsvm_model), node,
                            basename="LibSvmSvmr")

    def test_convert_nusvmr(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        prob = svmutil.svm_problem(y, X.tolist())        
                
        param = svmutil.svm_parameter()
        param.svm_type = NuSVR
        param.kernel_type = svmutil.RBF
        param.eps = 1
        if noprint:
            param.print_func = noprint

        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmNuSvmr", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIReg(libsvm_model), node,
                            basename="LibSvmNuSvmr")

    def test_convert_nusvmc(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = NuSVC
        param.kernel_type = svmutil.RBF
        param.eps = 1
        param.probability = 1
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmNuSvmc", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPIClProba2(libsvm_model), node,
                            basename="LibSvmNuSvmc-Dec2")

    def test_convert_svmc_linear_raw(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = SVC
        param.kernel_type = svmutil.LINEAR
        param.eps = 1
        param.probability = 0
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmcLinearRaw", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPICl(libsvm_model), node,
                            basename="LibSvmSvmcLinearRaw-Dec3", verbose=False,
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")

    def test_convert_svmc_raw(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = SVC
        param.kernel_type = svmutil.RBF
        param.eps = 1
        param.probability = 0
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmSvmcRaw", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        dump_data_and_model(X[:5].astype(numpy.float32), SkAPICl(libsvm_model), node,
                            basename="LibSvmSvmcRaw",
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")

    def test_convert_nusvmc_linear_raw(self):
        iris = load_iris()

        X = iris.data[:, :2]
        y = iris.target
        y[y == 2] = 1
        
        prob = svmutil.svm_problem(y, X.tolist())        

        param = svmutil.svm_parameter()
        param.svm_type = NuSVC
        param.kernel_type = svmutil.LINEAR
        param.eps = 1
        param.probability = 0
        if noprint:
            param.print_func = noprint
        
        libsvm_model = svmutil.svm_train(prob, param)

        node = convert(libsvm_model, "LibSvmNuSvmcRaw", [('input', FloatTensorType(shape=[1, 'None']))])
        self.assertTrue(node is not None)
        X2 = numpy.vstack([X[:5], X[60:65]])  # 5x0, 5x1
        dump_data_and_model(X2.astype(numpy.float32), SkAPICl(libsvm_model), node,
                            basename="LibSvmNuSvmcRaw", verbose=False,
                            allow_failure="StrictVersion(onnxruntime.__version__) <= StrictVersion('0.1.3')")


if __name__ == "__main__":
    unittest.main()
