# SPDX-License-Identifier: Apache-2.0

import unittest
import inspect
import os
import time
import pathlib
import numpy
import pandas
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml import buildInitialTypesSimple, buildInputDictSimple
from onnxmltools.utils.utils_backend import OnnxRuntimeAssertionError, compare_outputs
from onnxmltools.utils.utils_backend_onnxruntime import (
    run_with_runtime,
    _compare_expected,
)
from tests.sparkml import SparkMlTestCase


class ProfileSparkmlPipeline1(SparkMlTestCase):
    def _get_spark_options(self):
        # add additional jar files before creating SparkSession
        return {"spark.jars.packages": "ml.combust.mleap:mleap-spark_2.11:0.13.0"}


class ProfileSparkmlPipeline2(SparkMlTestCase):
    def test_profile_sparkml_pipeline(self):
        pass

        # add additional jar files before creating SparkSession
        this_script_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        input_path = os.path.join(
            this_script_dir, "data", "AdultCensusIncomeOriginal.csv"
        )
        full_data = (
            self.spark.read.format("csv")
            .options(header="true", inferschema="true")
            .load(input_path)
        )
        training_data, test_data = full_data.randomSplit([0.9, 0.1], seed=1)

        label = "income"
        dtypes = dict(training_data.dtypes)
        dtypes.pop(label)

        si_xvars = []
        ohe_xvars = []
        feature_cols = []
        for idx, key in enumerate(dtypes):
            if dtypes[key] == "string":
                feature_col = "-".join([key, "encoded"])
                feature_cols.append(feature_col)

                tmp_col = "-".join([key, "tmp"])
                si_xvars.append(
                    StringIndexer(inputCol=key, outputCol=tmp_col, handleInvalid="skip")
                )
                ohe_xvars.append(
                    OneHotEncoder(
                        inputCols=[tmp_col], outputCols=[feature_col], dropLast=False
                    )
                )
            else:
                feature_cols.append(key)
        si_label = StringIndexer(inputCol=label, outputCol="label")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        lr = LogisticRegression(regParam=0.001)
        pipeline = Pipeline(stages=si_xvars + ohe_xvars + [si_label, assembler, lr])

        # filter out the records which will cause error
        # use only one record for prediction
        test_data = test_data.limit(1)
        # create Spark and Onnx models
        model = pipeline.fit(training_data)
        model_onnx = convert_sparkml(
            model, "Sparkml Pipeline", buildInitialTypesSimple(test_data)
        )
        # save Onnx model for runtime usage
        if model_onnx is None:
            raise AssertionError("Failed to create the onnx model")
        model_path = os.path.join("tests", "profile_pipeline_model.onnx")
        with open(model_path, "wb") as f:
            f.write(model_onnx.SerializeToString())

        # Create MLeap model
        model_zip_path = os.path.join(this_script_dir, "tests", "mleap-pipeline.zip")
        if os.path.exists(model_zip_path):
            os.remove(model_zip_path)
        model_zip_url = "jar:" + pathlib.Path(model_zip_path).as_uri()
        # save the pipeline also in MLeap format
        empty_df = self.spark.createDataFrame([], model.transform(test_data).schema)
        model.serializeToBundle(model_zip_url, empty_df)
        mleap_pipeline = PipelineModel.deserializeFromBundle(model_zip_url)

        spark_times = []
        mleap_times = []
        runtime_times = []
        for i in range(0, 20):
            data_np = buildInputDictSimple(test_data)
            # run the model in Spark
            start = time.time()
            spark_prediction = model.transform(test_data)
            end = time.time()
            spark_times.append(1000 * (end - start))

            # run with MLeap
            start = time.time()
            mleap_prediction = mleap_pipeline.transform(test_data)
            end = time.time()
            mleap_times.append(1000 * (end - start))

            if i == 0:  # compare only once
                _compare_mleap_pyspark(mleap_prediction, spark_prediction)

            # run the model in onnx runtime
            start = time.time()
            output, session = run_with_runtime(data_np, model_path)
            end = time.time()
            runtime_times.append(1000 * (end - start))

            # compare results
            if i == 0:  # compare only once
                expected = [
                    spark_prediction.toPandas().label.values.astype(numpy.float32),
                    spark_prediction.toPandas().prediction.values.astype(numpy.float32),
                    spark_prediction.toPandas()
                    .probability.apply(lambda x: pandas.Series(x.toArray()))
                    .values.astype(numpy.float32),
                ]
                _compare_expected(
                    expected, output, session, model_path, decimal=5, onnx_shape=None
                )

        gen_plot(spark_times, mleap_times, runtime_times)


def _compare_mleap_pyspark(mleap_prediction, spark_prediction):
    spark_pandas = spark_prediction.toPandas()
    mleap_pandas = mleap_prediction.toPandas()
    spark_predicted_labels = spark_pandas.prediction.values
    mleap_predicted_labels = mleap_pandas.prediction.values
    msg = compare_outputs(spark_predicted_labels, mleap_predicted_labels, decimal=5)
    if msg:
        raise OnnxRuntimeAssertionError("Predictions in mleap and spark do not match")
    spark_probability = spark_pandas.probability.apply(
        lambda x: pandas.Series(x.toArray())
    ).values
    mleap_probability = mleap_pandas.probability.apply(
        lambda x: pandas.Series(x.toArray())
    ).values
    msg = compare_outputs(spark_probability, mleap_probability, decimal=5)
    if msg:
        raise OnnxRuntimeAssertionError("Probabilities in mleap and spark do not match")


def gen_plot(spark_times, mleap_times, runtime_times):
    import matplotlib.pyplot as pyplot

    pyplot.hist(spark_times, label="pyspark")
    pyplot.hist(mleap_times, label="MLeap")
    pyplot.hist(runtime_times, label="onnxruntime")
    pyplot.ylabel("Frequency")
    pyplot.xlabel("Prediction Time(ms)")
    pyplot.legend()
    fig = pyplot.gcf()
    # pyplot.show()
    pyplot.draw()
    fig.savefig("tests/spark-perf-histogram.png")


if __name__ == "__main__":
    unittest.main()
