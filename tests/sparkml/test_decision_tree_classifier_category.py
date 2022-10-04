# SPDX-License-Identifier: Apache-2.0

import sys
import os
import inspect
import unittest
import packaging.version as pv
import onnx
import pandas
import numpy
from sklearn.datasets import dump_svmlight_file
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.linalg import VectorUDT, SparseVector, Vectors
from pyspark.ml.feature import VectorAssembler
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import StringTensorType, FloatTensorType
from tests.sparkml.sparkml_test_utils import save_data_models, compare_results, run_onnx_model
from tests.sparkml import SparkMlTestCase
from pyspark.ml.feature import StringIndexer, VectorIndexer


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmDecisionTreeClassifierBig(SparkMlTestCase):

    # @unittest.skipIf(True, reason="Mismatched input dimensions.")
    @unittest.skipIf(sys.platform == 'win32',
                     reason="UnsatisfiedLinkError")
    @unittest.skipIf(sys.version_info < (3, 8),
                     reason="pickle fails on python 3.7")
    @unittest.skipIf(pv.Version(onnx.__version__) <= pv.Version('1.3'), 'Need Greater Opset 9')
    def test_tree_pipeline_category(self):

        features = numpy.random.randint(0, high=100, size=(10000, 50)) % 100
        features[:, :features.shape[1]-1] = features[:, :features.shape[1]-1] % 10
        features[:, :] *= numpy.random.randint(0, high=2, size=features.shape) % 2
        labels = (features.sum(axis=1) + features[:, -1] * 4)
        labels //= (labels.max() // 2 + 1)
        n_features = features.shape[1]
        df = pandas.DataFrame(features)
        features_names = [f"c{i}" for i in range(df.shape[1])]
        df.columns = features_names
        df["label"] = labels
        print(df.head())
        input_path = os.path.join(".", "sample_catdata.csv")
        df.to_csv(input_path, index=False)

        data = self.spark.read.csv(input_path, header=True, inferSchema=True)
        print(data.printSchema())
        va = VectorAssembler(inputCols=features_names, outputCol='features')
        va_df = va.transform(data)
        va_df = va_df.select(['features', 'label'])

        dt = DecisionTreeClassifier(labelCol="label", featuresCol='features', maxDepth=15)
        model = dt.fit(va_df)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Binary Class', [
            ('features', FloatTensorType([None, n_features]))
        ], spark_session=self.spark, target_opset=TARGET_OPSET)
        data_np = va_df.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(va_df)
        expected = [
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        paths = save_data_models(data_np, expected, model, model_onnx,
                                basename="SparkmlDecisionTreeBinaryClassCategory")
        onnx_model_path = paths[-1]
        output, output_shapes = run_onnx_model(['prediction', 'probability'], data_np, onnx_model_path)
        compare_results(expected, output, decimal=5)


if __name__ == "__main__":
    unittest.main()
