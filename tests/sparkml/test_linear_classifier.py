# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import inspect
import os
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.linalg import VectorUDT, SparseVector

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, run_onnx_model, compare_results
from tests.sparkml import SparkMlTestCase


class TestSparkmlLogisticRegression(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_logistic_regression_binary_class(self):
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        original_data = self.spark.read.format("libsvm").load(input_path)
        #
        # truncate the features
        #
        self.spark.udf.register("truncateFeatures", lambda x: SparseVector(5, range(0,5), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("label", "truncateFeatures(features) as features")
        lr = LogisticRegression(maxIter=100, tol=0.0001)
        model = lr.fit(data)
        # the name of the input for Logistic Regression is 'features'
        C = model.numFeatures
        model_onnx = convert_sparkml(model, 'sparkml logistic regression', [('features', FloatTensorType([1, C]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        # known error in onnxruntime 0.3.0 case
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlLogisticRegression")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_linear_svc(self):
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        original_data = self.spark.read.format("libsvm").load(input_path)
        #
        # truncate the features
        #
        self.spark.udf.register("truncateFeatures", lambda x: SparseVector(5, range(0,5), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("label", "truncateFeatures(features) as features")
        lsvc = LinearSVC(maxIter=10, regParam=0.01)
        model = lsvc.fit(data)
        # the name of the input for Logistic Regression is 'features'
        C = model.numFeatures
        model_onnx = convert_sparkml(model, 'Spark ML Linear SVC', [('features', FloatTensorType([1, C]))])
        self.assertTrue(model_onnx is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [ predicted.toPandas().prediction.values.astype(numpy.float32) ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                    basename="SparkmlLinearSVC")
        onnx_model_path = paths[3]
        output, output_shapes = run_onnx_model(['prediction'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

if __name__ == "__main__":
    unittest.main()
