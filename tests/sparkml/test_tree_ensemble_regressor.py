"""
Tests SparkML TreeEnsembleClassifier converter.
"""
import inspect
import unittest
import pandas
import numpy
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml import Pipeline

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType, StringTensorType
from sparkml import SparkMlTestCase, dump_data_and_sparkml_model
from pyspark.ml.feature import VectorIndexer, StringIndexer


class TestSparkmTreeEnsembleRegressor(SparkMlTestCase):
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

    def test_random_forrest_regression(self):
        import os
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        original_data = self.spark.read.format("libsvm").load(input_path)
        #
        # truncate the features
        #
        feature_count = 5
        self.spark.udf.register("truncateFeatures",
                                lambda x: SparseVector(feature_count, range(0,feature_count), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("cast(label as string) as label", "truncateFeatures(features) as features")
        label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                        maxCategories=10, handleInvalid='keep')

        rf = RandomForestRegressor(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml RandomForest Regressor', [
            ('label', StringTensorType([1, 1])),
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data)
        data_np = {
            'label': data.toPandas().label.values,
            'features': data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        }
        expected = [
            predicted.toPandas().indexedLabel.values.astype(numpy.int64),
            predicted.toPandas().prediction.values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlRandomForestRegressor")


if __name__ == "__main__":
    unittest.main()
