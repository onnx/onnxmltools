# SPDX-License-Identifier: Apache-2.0

"""
Mapping and utilities for the names of Params(propeties) that various Spark ML models
have for their input and output columns
"""
from .ops_names import get_sparkml_operator_name


def build_io_name_map():
    """
    map of spark models to input-output tuples
    Each lambda gets the corresponding input or output column name from the model
    """
    map = {
        "pyspark.ml.feature.BucketedRandomProjectionLSHModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.regression.AFTSurvivalRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.feature.ElementwiseProduct": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.MinHashLSHModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.Word2VecModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.IndexToString": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.ChiSqSelectorModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.classification.OneVsRestModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.regression.GBTRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.classification.GBTClassificationModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol"), "probability"],
        ),
        "pyspark.ml.feature.DCT": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.PCAModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.PolynomialExpansion": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.Tokenizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.classification.NaiveBayesModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [
                model.getOrDefault("predictionCol"),
                model.getOrDefault("probabilityCol"),
            ],
        ),
        "pyspark.ml.feature.VectorSlicer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.StopWordsRemover": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.NGram": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.Bucketizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.regression.RandomForestRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.classification.RandomForestClassificationModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [
                model.getOrDefault("predictionCol"),
                model.getOrDefault("probabilityCol"),
            ],
        ),
        "pyspark.ml.classification.MultilayerPerceptronClassificationModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [
                model.getOrDefault("predictionCol"),
                model.getOrDefault("probabilityCol"),
            ],
        ),
        "pyspark.ml.regression.DecisionTreeRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.classification.DecisionTreeClassificationModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [
                model.getOrDefault("predictionCol"),
                model.getOrDefault("probabilityCol"),
            ],
        ),
        "pyspark.ml.feature.VectorIndexerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.regression.GeneralizedLinearRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.regression.LinearRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.feature.ImputerModel": (
            lambda model: (
                model.getOrDefault("inputCols")
                if model.isSet("inputCols")
                else [model.getOrDefault("inputCol")]
            ),
            lambda model: (
                model.getOrDefault("outputCols")
                if model.isSet("outputCols")
                else [model.getOrDefault("outputCol")]
            ),
        ),
        "pyspark.ml.feature.MaxAbsScalerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.MinMaxScalerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.StandardScalerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.Normalizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.Binarizer": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.feature.CountVectorizerModel": (
            lambda model: [model.getOrDefault("inputCol")],
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.classification.LinearSVCModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
        "pyspark.ml.classification.LogisticRegressionModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [
                model.getOrDefault("predictionCol"),
                model.getOrDefault("probabilityCol"),
            ],
        ),
        "pyspark.ml.feature.OneHotEncoderModel": (
            lambda model: (
                model.getOrDefault("inputCols")
                if model.isSet("inputCols")
                else [model.getOrDefault("inputCol")]
            ),
            lambda model: (
                model.getOrDefault("outputCols")
                if model.isSet("outputCols")
                else [model.getOrDefault("outputCol")]
            ),
        ),
        "pyspark.ml.feature.StringIndexerModel": (
            lambda model: (
                model.getOrDefault("inputCols")
                if model.isSet("inputCols")
                else [model.getOrDefault("inputCol")]
            ),
            lambda model: (
                model.getOrDefault("outputCols")
                if model.isSet("outputCols")
                else [model.getOrDefault("outputCol")]
            ),
        ),
        "pyspark.ml.feature.VectorAssembler": (
            lambda model: model.getOrDefault("inputCols"),
            lambda model: [model.getOrDefault("outputCol")],
        ),
        "pyspark.ml.clustering.KMeansModel": (
            lambda model: [model.getOrDefault("featuresCol")],
            lambda model: [model.getOrDefault("predictionCol")],
        ),
    }
    return map


io_name_map = build_io_name_map()


def get_input_names(model):
    """
    Returns the name(s) of the input(s) for a SparkML operator
    :param model: SparkML Model
    :return: list of input names
    """
    return io_name_map[get_sparkml_operator_name(type(model))][0](model)


def get_output_names(model):
    """
    Returns the name(s) of the output(s) for a SparkML operator
    :param model: SparkML Model
    :return: list of output names
    """
    return io_name_map[get_sparkml_operator_name(type(model))][1](model)
