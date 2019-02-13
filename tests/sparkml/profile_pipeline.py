from onnxmltools.utils import dump_data_and_sparkml_model, start_spark, stop_spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler

from onnxmltools import convert_sparkml
from onnxmltools.convert.sparkml import buildInitialTypesSimple, buildInputDictSimple
from onnxmltools.utils.utils_backend_onnxruntime import run_with_runtime


def profile_sparkml_pipeline():
    import inspect
    import os
    import numpy
    import pandas
    import time
    spark = start_spark()
    this_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    input_path = os.path.join(this_script_dir, "data", "AdultCensusIncomeOriginal.csv")
    full_data = spark.read.format('csv') \
        .options(header='true', inferschema='true').load(input_path)
    cols = ['workclass', 'education', 'marital_status']
    training_data, test_data = full_data.select('income', *cols).limit(10000).randomSplit([0.9, 0.1], seed=1)

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
            si_xvars.append(StringIndexer(inputCol=key, outputCol=tmp_col, handleInvalid="error"))
            ohe_xvars.append(OneHotEncoderEstimator(inputCols=[tmp_col], outputCols=[feature_col], dropLast=False))
        else:
            feature_cols.append(key)
    si_label = StringIndexer(inputCol=label, outputCol='label')
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LogisticRegression(regParam=0.001)
    pipeline = Pipeline(stages=[*si_xvars, *ohe_xvars, si_label, assembler, lr])

    model = pipeline.fit(training_data)
    model_onnx = convert_sparkml(model, 'Sparkml Pipeline', buildInitialTypesSimple(test_data))
    if model_onnx is None: raise AssertionError("Failed to create the onnx model")
    model_path = os.path.join("tests", "profile_pipeline_model.onnx")
    with open(model_path, "wb") as f:
        f.write(model_onnx.SerializeToString())

    columns = [ 'input_rec_count', 'pyspark (sec)', 'onnxruntime (sec)']
    rec_counts = []
    spark_times = []
    runtime_times = []
    for i in range(0, 8):
        rec_counts.append(test_data.count())
        # run the model
        data_np = buildInputDictSimple(test_data)
        # run the model
        start = time.time()
        predicted = model.transform(test_data)
        pred_count = predicted.count()
        end = time.time()
        spark_times.append(1000*(end - start))

        # test for correcness also
        expected = [
            predicted.toPandas().label.values.astype(numpy.float32),
            predicted.toPandas().prediction.values.astype(numpy.float32),
            predicted.toPandas().probability.apply(lambda x: pandas.Series(x.toArray())).values.astype(numpy.float32)
        ]
        start = time.time()
        run_with_runtime(data_np, model_path)
        end = time.time()
        runtime_times.append(1000*(end - start))

        # each time in this loop double the number of rows
        test_data = test_data.union(test_data)

    results = pandas.DataFrame(data={
        'input_rec_count': rec_counts,
        'pyspark (ms)': spark_times,
        'onnxruntime (ms)': runtime_times
    })
    print(results)
    stop_spark(spark)

if __name__ == "__main__":
    profile_sparkml_pipeline()
