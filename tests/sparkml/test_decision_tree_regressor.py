import sys
import inspect
import unittest
from distutils.version import StrictVersion

import onnx
import pandas
import numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml import SparkMlTestCase, dump_data_and_sparkml_model
from pyspark.ml.feature import VectorIndexer


class TestSparkmTreeEnsembleRegressor(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
    def test_decision_tree_regressor_pipeline(self):
        import os
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        data = self.spark.read.format("libsvm").load(input_path)

        featureIndexer = \
            VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4, handleInvalid='keep')
        feature_count = data.select('features').first()[0].size
        (trainingData, testData) = data.randomSplit([0.7, 0.3])
        dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
        pipeline = Pipeline(stages=[featureIndexer, dt])
        model = pipeline.fit(trainingData)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Regressor Pipeline', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(testData)
        data_np = testData.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreeRegressorPipeline")

    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_decision_tree_regressor(self):
        features = [[0, 1], [1, 1], [2, 0]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [100, -10, 50]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeRegressor(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        feature_count = data.select('features').first()[0].size
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Regressor', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreeRegressor")


if __name__ == "__main__":
    unittest.main()
