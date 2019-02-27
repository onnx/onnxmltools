import unittest
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml import buildInitialTypesSimple, buildInputDictSimple
from onnxmltools.utils.utils_backend_onnxruntime import run_with_runtime, _compare_expected
from tests.sparkml import SparkMlTestCase


class ProfileSparkmlPipeline(SparkMlTestCase):
    @unittest.skipIf(sys.version_info[0] == 2, reason="Sparkml not tested on python 2")
    def test_profile_sparkml_pipeline(self):
        import inspect
        import os
        import numpy
        import pandas
        import time
        this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        input_path = os.path.join(this_script_dir, "data", "AdultCensusIncomeOriginal.csv")
        full_data = self.spark.read.format('csv') \
            .options(header='true', inferschema='true').load(input_path)
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
                si_xvars.append(StringIndexer(inputCol=key, outputCol=tmp_col, handleInvalid="skip"))
                ohe_xvars.append(OneHotEncoderEstimator(inputCols=[tmp_col], outputCols=[feature_col], dropLast=False))
            else:
                feature_cols.append(key)
        si_label = StringIndexer(inputCol=label, outputCol='label')
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        lr = LogisticRegression(regParam=0.001)
        pipeline = Pipeline(stages=si_xvars + ohe_xvars + [ si_label, assembler, lr])

        # filter out the records which will cause error
        model = pipeline.fit(training_data)
        model_onnx = convert_sparkml(model, 'Sparkml Pipeline', buildInitialTypesSimple(test_data))
        if model_onnx is None: raise AssertionError("Failed to create the onnx model")
        model_path = os.path.join("tests", "profile_pipeline_model.onnx")
        with open(model_path, "wb") as f:
            f.write(model_onnx.SerializeToString())

        rec_counts = []
        spark_times = []
        runtime_times = []
        for i in range(0, 4):
            rec_counts.append(test_data.count())
            data_np = buildInputDictSimple(test_data)
            # run the model in Spark
            start = time.time()
            predicted = model.transform(test_data)
            end = time.time()
            spark_times.append(1000*(end - start))

            # test for correctness also
            expected = [
                predicted.toPandas().label.values.astype(numpy.float32),
                predicted.toPandas().prediction.values.astype(numpy.float32),
                predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
            ]
            # run the model in onnx runtime
            start = time.time()
            output, session = run_with_runtime(data_np, model_path)
            end = time.time()
            runtime_times.append(1000*(end - start))

            # compare results
            _compare_expected(expected, output, session, model_path, decimal=5, onnx_shape=None)

            # each time in this loop double the number of rows
            test_data = test_data.union(test_data)

        results = pandas.DataFrame(data={
            'input_rec_count': rec_counts,
            'pyspark (ms)': spark_times,
            'onnxruntime (ms)': runtime_times
        })
        print(results)

if __name__ == "__main__":
    unittest.main()

