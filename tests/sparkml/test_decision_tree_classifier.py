# SPDX-License-Identifier: Apache-2.0

import sys
import inspect
import unittest
from distutils.version import StrictVersion
import onnx
import pandas
import numpy
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.linalg import VectorUDT, SparseVector, Vectors
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, compare_results, run_onnx_model
from tests.sparkml import SparkMlTestCase
from pyspark.ml.feature import StringIndexer, VectorIndexer


class TestSparkmDecisionTreeClassifier(SparkMlTestCase):

    @unittest.skipIf(True, reason="Mismatched input dimensions.")
    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    @unittest.skipIf(StrictVersion(onnx.__version__) <= StrictVersion('1.3'), 'Need Greater Opset 9')
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
                                lambda x: SparseVector(feature_count, range(0, feature_count), x.toArray()[125:130]),
                                VectorUDT())
        data = original_data.selectExpr("cast(label as string) as label", "truncateFeatures(features) as features")
        label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid='error')
        feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures",
                                        maxCategories=10, handleInvalid='error')

        dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
        pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
        model = pipeline.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Pipeline', [
            ('label', StringTensorType([None, 1])),
            ('features', FloatTensorType([None, feature_count]))
        ], spark_session=self.spark)
        self.assertTrue(model_onnx is not None)
        # run the model
        predicted = model.transform(data.limit(1))
        data_np = {
            'label': data.limit(1).toPandas().label.values.reshape((-1, 1)),
            'features': data.limit(1).toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        }
        expected = [
            predicted.toPandas().indexedLabel.values.astype(numpy.int64),
            predicted.toPandas().prediction.values.astype(numpy.int64),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlDecisionTreePipeline")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['indexedLabel', 'prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(True, reason="Mismatched input dimensions.")
    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_tree_one_class_classification(self):
        features = [[0., 1.], [1., 1.], [2., 0.]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [1, 1, 1]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree One Class', [
            ('features', FloatTensorType([None, 2]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlDecisionTreeBinaryClass")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_tree_binary_classification(self):
        features = [[0, 1], [1, 1], [2, 0]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [0, 1, 0]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Binary Class', [
            ('features', FloatTensorType([None, 2]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlDecisionTreeBinaryClass")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)

    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    def test_tree_multiple_classification(self):
        features = [[0, 1], [1, 1], [2, 0], [0.5, 0.5], [1.1, 1.1], [2.1, 0.1]]
        features = numpy.array(features, dtype=numpy.float32)
        labels = [0, 1, 2, 1, 1, 2]
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
        model = dt.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Multi Class', [
            ('features', FloatTensorType([None, 2]))
        ], spark_session=self.spark)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlDecisionTreeMultiClass")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
