"""
Tests SparkML TreeEnsembleClassifier converter.
"""
import inspect
import unittest

from pyspark.ml import Pipeline
from  pyspark.ml.classification import DecisionTreeClassifier

from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
from sparkml import SparkMlTestCase
from pyspark.ml.feature import StringIndexer, VectorIndexer


class TestSparkmTreeEnsembleClassifier(SparkMlTestCase):
    def test_model_tree_ensemble_classifier(self):
        import os
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "sample_libsvm_data.txt")
        data = self.spark.read.format("libsvm").load(input_path)

        labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
        featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
        model = pipeline.fit(trainingData)
        # C = model.numFeatures
        model_onnx = convert_sparkml(model, 'Sparkml Pipeline', [
            ('label', StringTensorType([1, 1])),
            ('features', FloatTensorType([1, 2]))
        ], spark_session=self.spark)



if __name__ == "__main__":
    unittest.main()
