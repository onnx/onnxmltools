"""
Tests SparkML LogisticRegression converter.
"""
import unittest
import numpy
import sys
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import VectorUDT, SparseVector

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model


class TestSparkmlLogisticRegression(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_model_logistic_regression_binary_class(self):
        import inspect
        import os
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
        model_onnx = convert_sparkml(model, 'sparkml logistic regression', [('features', FloatTensorType([1, model.numFeatures]))])
        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)
        # run the model
        import pandas
        predicted = model.transform(data)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlLogisticRegression")


if __name__ == "__main__":
    unittest.main()
