# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import numpy
import pandas
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
from pyspark.ml import Pipeline
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from tests.sparkml.sparkml_test_utils import (
    save_data_models,
    run_onnx_model,
    compare_results,
)
from tests.sparkml import SparkMlTestCase


TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())


class TestSparkmlKMeansModel(SparkMlTestCase):
    @unittest.skipIf(sys.version_info < (3, 8), reason="pickle fails on python 3.7")
    def test_model_k_means_euclidean(self):
        """
        Testing ONNX conversion for Spark KMeansModel
        when distanceMeasure is set to "euclidean".
        """
        kmeans_euclidean = KMeans(
            k=3,
            distanceMeasure="euclidean",
            featuresCol="features_euclidean",
            predictionCol="prediction_euclidean",
        )
        kmeans_cosine = KMeans(
            k=3,
            distanceMeasure="cosine",
            featuresCol="features_cosine",
            predictionCol="prediction_cosine",
        )

        data = self.spark.createDataFrame(
            [
                (
                    0,
                    Vectors.dense([1.0, 3.1, -1.0]),
                    Vectors.dense([1.0, 1.0, 1.0]),
                ),
                (
                    1,
                    Vectors.dense([1.1, 3.0, -1.1]),
                    Vectors.dense([2.0, 2.0, 2.0]),
                ),
                (
                    2,
                    Vectors.dense([-3.0, 5.1, 9.0]),
                    Vectors.dense([-1.0, 3.0, -5.0]),
                ),
                (
                    3,
                    Vectors.dense([-2.9, 4.9, 8.9]),
                    Vectors.dense([-2.0, 6.0, -10.0]),
                ),
                (
                    4,
                    Vectors.dense([5.0, -3.5, 2.0]),
                    Vectors.dense([1.0, -2.0, 4.0]),
                ),
                (
                    5,
                    Vectors.dense([5.1, -3.3, 2.1]),
                    Vectors.dense([2.0, -4.0, 8.0]),
                ),
            ],
            ["id", "features_euclidean", "features_cosine"],
        )

        model = Pipeline(stages=[kmeans_euclidean, kmeans_cosine]).fit(data)
        model_onnx = convert_sparkml(
            model,
            "Sparkml KMeansModel",
            [
                ("features_euclidean", FloatTensorType([None, 3])),
                ("features_cosine", FloatTensorType([None, 3])),
            ],
            target_opset=TARGET_OPSET,
        )

        self.assertTrue(model_onnx is not None)
        self.assertTrue(model_onnx.graph.node is not None)

        # run the model
        predicted = model.transform(data).toPandas()

        data_pd = data.toPandas()
        data_np = {
            "features_euclidean": data_pd.features_euclidean.apply(
                lambda x: pandas.Series(x.toArray())
            ).values.astype(numpy.float32),
            "features_cosine": data_pd.features_cosine.apply(
                lambda x: pandas.Series(x.toArray())
            ).values.astype(numpy.float32),
        }

        expected = {
            "prediction_euclidean": numpy.asarray(
                predicted.prediction_euclidean.values
            ),
            "prediction_cosine": numpy.asarray(predicted.prediction_cosine.values),
        }

        paths = save_data_models(
            data_np, expected, model, model_onnx, basename="SparkmlKMeansModel"
        )
        onnx_model_path = paths[-1]

        output_names = ["prediction_euclidean", "prediction_cosine"]
        output, output_shapes = run_onnx_model(output_names, data_np, onnx_model_path)
        actual_output = dict(zip(output_names, output))

        assert output_shapes[0] == [None]
        assert output_shapes[1] == [None]
        compare_results(
            expected["prediction_euclidean"],
            actual_output["prediction_euclidean"],
            decimal=5,
        )
        compare_results(
            expected["prediction_cosine"], actual_output["prediction_cosine"], decimal=5
        )


if __name__ == "__main__":
    unittest.main()
