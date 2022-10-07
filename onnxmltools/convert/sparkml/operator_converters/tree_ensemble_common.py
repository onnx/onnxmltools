# SPDX-License-Identifier: Apache-2.0

import tempfile
import os
import time
import numpy
from pyspark.sql import SparkSession

class SparkMLTree(dict):
    pass


def sparkml_tree_dataset_to_sklearn(tree_df, is_classifier):
    feature = []
    threshold = []
    tree_pandas = tree_df.toPandas()
    children_left = tree_pandas.leftChild.values.tolist()
    children_right = tree_pandas.rightChild.values.tolist()
    ids = tree_pandas.id.values.tolist()
    if is_classifier:
        value = numpy.array(tree_pandas.impurityStats.values.tolist())
    else:
        value = tree_pandas.prediction.values.tolist()

    for item in tree_pandas.split:
        if isinstance(item, dict):
            try:
                feature.append(item["featureIndex"])
                threshold.append(item["leftCategoriesOrThreshold"])
            except KeyError as e:
                raise RuntimeError(f"Unable to process {item}.")
        else:
            tuple_item = tuple(item)
            feature.append(item[0])
            threshold.append(item[1][0] if len(item[1]) >= 1 else -1.0)

    tree = SparkMLTree()
    tree.nodes_ids = ids
    tree.children_left = children_left
    tree.children_right = children_right
    tree.value = numpy.asarray(value, dtype=numpy.float32)
    tree.feature = feature
    tree.threshold = threshold
    tree.node_count = tree_df.count()
    return tree


def save_read_sparkml_model_data(spark: SparkSession, model):
    tdir = tempfile.tempdir
    if tdir is None:
        local_dir = spark._jvm.org.apache.spark.util.Utils.getLocalDir(spark._jsc.sc().conf())
        tdir = spark._jvm.org.apache.spark.util.Utils.createTempDir(local_dir, "onnx").getAbsolutePath()
    if tdir is None:
        raise FileNotFoundError(
            "Unable to create a temporary directory for model '{}'"
            ".".format(type(model).__name__))
    path = os.path.join(tdir, type(model).__name__ + "_" + str(time.time()))
    model.write().overwrite().save(path)
    df = spark.read.parquet(os.path.join(path, 'data'))
    return df
