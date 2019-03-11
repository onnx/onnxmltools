"""
Tests SparkML TreeEnsembleClassifier converter.
"""
import inspect
import unittest
import pandas
import numpy
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.linalg import VectorUDT, SparseVector, Vectors

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
from sparkml import SparkMlTestCase, dump_data_and_sparkml_model
from pyspark.ml.feature import StringIndexer, VectorIndexer


class TestSparkmTreeEnsembleClassifier(SparkMlTestCase):
    def test_tree_pipeline(self):
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

        dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Pipeline', [
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
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreePipeline")

    def test_tree_one_class_classification(self):
        features = [[0., 1.], [1., 1.], [2., 0.]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [1, 1, 1]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        feature_count = 1
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree One Class', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreeOneClass")

    def test_tree_binary_classification(self):
        features = [[0, 1], [1, 1], [2, 0]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [0, 1, 0]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        feature_count = 2
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Binary Class', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreeBinaryClass")

    def test_tree_multiple_classification(self):
        features = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [0, 1, 2, 1, 1, 2]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        feature_count = 2
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Multi Class', [
            ('features', FloatTensorType([1, feature_count]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreeMultiClass")

    def test_random_forrest_classification(self):
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

        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Pipeline', [
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
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        dump_data_and_sparkml_model(data_np, expected, model, model_onnx,
                                    basename="SparkmlDecisionTreePipeline")


if __name__ == "__main__":
    unittest.main()
