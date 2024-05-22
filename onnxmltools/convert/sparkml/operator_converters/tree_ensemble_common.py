# SPDX-License-Identifier: Apache-2.0

import tempfile
import os
import time
import numpy
import re
from pyspark.sql import SparkSession


class SparkMLTree(dict):
    pass


def sparkml_tree_dataset_to_sklearn(tree_df, is_classifier):
    feature = []
    threshold = []
    tree_pandas = tree_df.toPandas().sort_values("id")
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
                threshold.append(
                    item["leftCategoriesOrThreshold"][0]
                    if len(item["leftCategoriesOrThreshold"]) >= 1
                    else -1.0
                )
            except KeyError:
                raise RuntimeError(f"Unable to process {item}.")
        else:
            tuple(item)
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
    # Get the value of spark.master
    spark_mode = spark.conf.get("spark.master")

    # Check the value of spark.master using regular expression
    if "spark://" in spark_mode and (
        "localhost" not in spark_mode or "127.0.0.1" not in spark_mode
    ):
        dfs_key = "ONNX_DFS_PATH"
        try:
            dfs_path = spark.conf.get("ONNX_DFS_PATH")
        except Exception:
            raise ValueError(
                "Configuration property '{}' does not exist for SparkSession. \
                Please set this variable to a root distributed file system path to allow \
                for saving and reading of spark models in cluster mode. \
                You can set this in your SparkConfig \
                by setting sparkBuilder.config(ONNX_DFS_PATH, dfs_path)".format(
                    dfs_key
                )
            )
        if dfs_path is None:
            # If dfs_path is not specified, throw an error message
            # dfs_path arg is required for cluster mode
            raise ValueError(
                "Argument dfs_path is required for saving model '{}' in cluster mode. \
                You can set this in your SparkConfig by \
                setting sparkBuilder.config(ONNX_DFS_PATH, dfs_path)".format(
                    type(model).__name__
                )
            )
        else:
            # Check that the dfs_path is a valid distributed file system path
            # This can be hdfs, wabs, s3, etc.
            if re.match(r"^[a-zA-Z]+://", dfs_path) is None:
                raise ValueError(
                    "Argument dfs_path '{}' is not a valid distributed path".format(
                        dfs_path
                    )
                )
            else:
                # If dfs_path is specified, save the model to a tmp directory
                # The dfs_path will be the root of the /tmp
                tdir = os.path.join(dfs_path, "tmp/onnx")
    else:
        # If spark.master is not set or set to local, save the model to a local path.
        tdir = tempfile.tempdir
        if tdir is None:
            local_dir = spark._jvm.org.apache.spark.util.Utils.getLocalDir(
                spark._jsc.sc().conf()
            )
            tdir = spark._jvm.org.apache.spark.util.Utils.createTempDir(
                local_dir, "onnx"
            ).getAbsolutePath()
        if tdir is None:
            raise FileNotFoundError(
                "Unable to create a temporary directory for model '{}'"
                ".".format(type(model).__name__)
            )

    path = os.path.join(tdir, type(model).__name__ + "_" + str(time.time()))
    model.write().overwrite().save(path)
    df = spark.read.parquet(os.path.join(path, "data"))
    return df


def get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["class_treeids"] = []
    attrs["class_nodeids"] = []
    attrs["class_ids"] = []
    attrs["class_weights"] = []
    return attrs


def get_default_tree_regressor_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["n_targets"] = 0
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["target_treeids"] = []
    attrs["target_nodeids"] = []
    attrs["target_ids"] = []
    attrs["target_weights"] = []
    return attrs


def add_node(
    attr_pairs,
    is_classifier,
    tree_id,
    tree_weight,
    node_id,
    feature_id,
    mode,
    value,
    true_child_id,
    false_child_id,
    weights,
    weight_id_bias,
    leaf_weights_are_counts,
):
    attr_pairs["nodes_treeids"].append(tree_id)
    attr_pairs["nodes_nodeids"].append(node_id)
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(mode)
    attr_pairs["nodes_values"].append(value)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_falsenodeids"].append(false_child_id)
    attr_pairs["nodes_missing_value_tracks_true"].append(False)
    attr_pairs["nodes_hitrates"].append(1.0)

    # Add leaf information for making prediction
    if mode == "LEAF":
        flattened_weights = weights.flatten()
        factor = tree_weight
        # If the values stored at leaves are counts of possible classes,
        # we need convert them to probabilities by
        # doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            factor /= float(s) if s != 0.0 else 1.0
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]

        # Note that attribute names for making prediction
        # are different for classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs["class_treeids"].append(tree_id)
                attr_pairs["class_nodeids"].append(node_id)
                attr_pairs["class_ids"].append(i + weight_id_bias)
                attr_pairs["class_weights"].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs["target_treeids"].append(tree_id)
                attr_pairs["target_nodeids"].append(node_id)
                attr_pairs["target_ids"].append(i + weight_id_bias)
                attr_pairs["target_weights"].append(w)


def add_tree_to_attribute_pairs(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
):
    for i in range(tree.node_count):
        node_id = tree.nodes_ids[i]
        weight = tree.value[i]

        if tree.children_left[i] >= 0 or tree.children_right[i] >= 0:
            mode = "BRANCH_LEQ"
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
        else:
            mode = "LEAF"
            feat_id = 0
            threshold = 0.0
            left_child_id = 0
            right_child_id = 0

        add_node(
            attr_pairs,
            is_classifier,
            tree_id,
            tree_weight,
            node_id,
            feat_id,
            mode,
            threshold,
            left_child_id,
            right_child_id,
            weight,
            weight_id_bias,
            leaf_weights_are_counts,
        )
