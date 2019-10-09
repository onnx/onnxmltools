import sys
import unittest
import numpy
import inspect
import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlLinearRegression(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_linear_regression_basic(self):
        data = self.spark.createDataFrame([
            (1.0, 2.0, Vectors.dense(1.0)),
            (0.0, 2.0, Vectors.sparse(1, [], []))
        ], ["label", "weight", "features"])
        lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal", weightCol="weight")
        model = lr.fit(data)
        # the name of the input is 'features'
        C = model.numFeatures
        model_onnx = convert_sparkml(model, 'sparkml LinearRegressorBasic', [('features', FloatTensorType([None, C]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [ predicted.toPandas().prediction.values.astype(numpy.float32) ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlLinearRegressor_Basic")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_linear_regression(self):
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_linear_regression_data.txt")
        data = self.spark.read.format("libsvm").load(input_path)

        lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
        model = lr.fit(data)
        # the name of the input is 'features'
        C = model.numFeatures
        model_onnx = convert_sparkml(model, 'sparkml LinearRegressor', [('features', FloatTensorType([None, C]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [ predicted.toPandas().prediction.values.astype(numpy.float32) ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlLinearRegressor")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_generalized_linear_regression(self):
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_linear_regression_data.txt")
        data = self.spark.read.format("libsvm").load(input_path)

        lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
        model = lr.fit(data)
        # the name of the input is 'features'
        C = model.numFeatures
        model_onnx = convert_sparkml(model, 'sparkml GeneralizedLinearRegression', [('features', FloatTensorType([None, C]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [ predicted.toPandas().prediction.values.astype(numpy.float32) ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlGeneralizedLinearRegression")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
