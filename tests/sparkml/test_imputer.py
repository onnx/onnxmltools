# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
from pyspark.ml.feature import Imputer
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase

## For some reason during the spark bring up and shutdown something happens causing Imputer
## tests to fail. For that you need to run each test here individually
## for now these will be commented out so as not to break the build
##      AttributeError: 'NoneType' object has no attribute 'setCallSite' on model.surrogateDF
##  Therefore we leave these tests out for now until a newere version of pyspark is availabe that address this issue
class TestSparkmlImputer(SparkMlTestCase):

    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_imputer_single(self):
        self._imputer_test_single()

    @unittest.skipIf(True, reason="Name:'Split' Status Message: Cannot split using values in 'split")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_imputer_multi(self):
        self._imputer_test_multi()

    def _imputer_test_multi(self):
        data = self.spark.createDataFrame([
            (1.0, float("nan")),
            (2.0, float("nan")),
            (float("nan"), 3.0),
            (4.0, 4.0),
            (5.0, 5.0)
        ], ["a", "b"])
        imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
        model = imputer.fit(data)
    
        # the input name should match the inputCols above
        model_onnx = convert_sparkml(model, 'Sparkml Imputer Multi Input', [
            ('a', FloatTensorType([None, 1])),
            ('b', FloatTensorType([None, 1]))])
        self.assertTrue(model_onnx is not None)
    
        # run the model
        predicted = model.transform(data)
        expected = predicted.select("out_a", "out_b").toPandas().values.astype(numpy.float32)
        data_np = data.toPandas().values.astype(numpy.float32)
        data_np = {'a': data_np[:, :1], 'b': data_np[:, 1:]}
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlImputerMulti")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['out_a', 'out_b'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)
    
    def _imputer_test_single(self):
        data = self.spark.createDataFrame([
            (1.0, float("nan")),
            (2.0, float("nan")),
            (float("nan"), 3.0),
            (4.0, 4.0),
            (5.0, 5.0)
        ], ["a", "b"])
        imputer = Imputer(inputCols=["a"], outputCols=["out_a"])
        model = imputer.fit(data)
    
        # the input name should match the inputCols above
        model_onnx = convert_sparkml(model, 'Sparkml Imputer', [
            ('a', FloatTensorType([None, 1]))])
        self.assertTrue(model_onnx is not None)
    
        # run the model
        predicted = model.transform(data)
        expected = predicted.select("out_a").toPandas().values.astype(numpy.float32)
        data_np = data.toPandas().a.values.astype(numpy.float32)
        data_np = data_np.reshape((-1, 1))
        paths = save_data_models(data_np, expected, model, model_onnx, basename="SparkmlImputerSingle")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['out_a'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
