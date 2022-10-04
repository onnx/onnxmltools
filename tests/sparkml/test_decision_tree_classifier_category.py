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

        features = numpy.random.randint(0, high=100, size=(10000, 10)) % 100
        features[:, :features.shape[1]-1] = features[:, :features.shape[1]-1] % 10
        features[:, :] *= numpy.random.randint(0, high=2, size=features.shape) % 2
        labels = features[:, -1] % 2
        n_features = features.shape[1]
        features = features.tolist()
        labels = labels.tolist()
        #input_path = os.path.join(this_script_dir, "sample_libsvm_catdata.svm")
        #dump_svmlight_file(mat, label, input_path)

        #this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        #original_data = self.spark.read.format("libsvm").load(input_path)
        dd = [(labels[i], Vectors.dense(features[i])) for i in range(len(labels))]
        data = self.spark.createDataFrame(self.spark.sparkContext.parallelize(dd), schema=["label", "features"])
        dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)
        model = dt.fit(data)
        model_onnx = convert_sparkml(model, 'Sparkml Decision Tree Binary Class', [
            ('features', FloatTensorType([None, n_features]))
        ], spark_session=self.spark, target_opset=TARGET_OPSET)
        data_np = data.toPandas().features.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        predicted = model.transform(data)
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
