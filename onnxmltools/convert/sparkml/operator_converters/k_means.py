from ...common._registration import register_converter, register_shape_calculator
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._topology import Operator, Scope, ModelComponentContainer
from ....proto import onnx_proto
from pyspark.ml.clustering import KMeansModel
import numpy as np


def convert_sparkml_k_means_model(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    if container.target_opset < 7:
        raise NotImplementedError(
            "Converting to ONNX for KMeansModel is not supported in opset < 7"
        )

    op: KMeansModel = operator.raw_operator
    centers: np.ndarray = np.vstack(op.clusterCenters())

    K = centers.shape[0]  # number of clusters
    C = operator.inputs[0].type.shape[1]  # Number of features from input

    if centers.shape[1] != C:
        raise ValueError(
            f"Number of features {centers.shape[1]} "
            f"in input does not match number of features in centers {C}"
        )

    # [K x C]
    centers_variable_name = scope.get_unique_variable_name("centers")
    container.add_initializer(
        centers_variable_name,
        onnx_proto.TensorProto.FLOAT,
        centers.shape,
        centers.flatten().astype(np.float32),
    )

    distance_output_variable_name = scope.get_unique_variable_name("distance_output")

    if op.getDistanceMeasure() == "euclidean":
        # [1 x K]
        centers_row_squared_sum_variable_name = scope.get_unique_variable_name(
            "centers_row_squared_sum"
        )
        centers_row_squared_sum = (
            np.sum(centers**2, axis=-1).flatten().astype(np.float32)
        )
        container.add_initializer(
            centers_row_squared_sum_variable_name,
            onnx_proto.TensorProto.FLOAT,
            [1, K],
            centers_row_squared_sum,
        )

        # input_row_squared_sum: [N x 1]
        input_row_squared_sum_variable_name = scope.get_unique_variable_name(
            "input_row_squared_sum"
        )
        reduce_sum_square_attrs = {
            "name": scope.get_unique_operator_name("input_row_squared_sum"),
            "axes": [1],
            "keepdims": 1,
        }
        container.add_node(
            op_type="ReduceSumSquare",
            inputs=[operator.inputs[0].full_name],
            outputs=[input_row_squared_sum_variable_name],
            **reduce_sum_square_attrs,
        )

        # -2 * input * Transpose(Center) + input_row_squared_sum: [N x K]
        gemm_output_variable_name = scope.get_unique_variable_name("gemm_output")
        gemm_attrs = {
            "name": scope.get_unique_operator_name("GeMM"),
            "alpha": -2.0,
            "beta": 1.0,
            "transB": 1,
        }
        container.add_node(
            op_type="Gemm",
            inputs=[
                operator.inputs[0].full_name,
                centers_variable_name,
                input_row_squared_sum_variable_name,
            ],
            outputs=[gemm_output_variable_name],
            op_version=7,
            **gemm_attrs,
        )

        # Euclidean Distance Squared = input_row_squared_sum - 2 *
        # input * Transpose(Center) + Transpose(centers_row_squared_sum)
        # [N x K]
        container.add_node(
            op_type="Add",
            inputs=[gemm_output_variable_name, centers_row_squared_sum_variable_name],
            outputs=[distance_output_variable_name],
            op_version=7,
        )
    elif op.getDistanceMeasure() == "cosine":
        # centers_row_norm2: [1 x K]
        centers_row_norm2_variable_name = scope.get_unique_variable_name(
            "centers_row_norm2"
        )
        centers_row_norm2 = (
            np.linalg.norm(centers, ord=2, axis=1).flatten().astype(np.float32)
        )
        container.add_initializer(
            centers_row_norm2_variable_name,
            onnx_proto.TensorProto.FLOAT,
            [1, K],
            centers_row_norm2,
        )

        # input_row_norm2: [N x 1]
        input_row_norm2_variable_name = scope.get_unique_variable_name(
            "input_row_norm2"
        )
        reduce_l2_attrs = {
            "name": scope.get_unique_operator_name("input_row_norm2"),
            "axes": [1],
            "keepdims": 1,
        }
        container.add_node(
            op_type="ReduceL2",
            inputs=[operator.inputs[0].full_name],
            outputs=[input_row_norm2_variable_name],
            **reduce_l2_attrs,
        )

        # input * Transpose(Center): [N x K]
        zeros_variable_name = scope.get_unique_variable_name("zeros")
        container.add_initializer(
            zeros_variable_name,
            onnx_proto.TensorProto.FLOAT,
            [1, K],
            np.zeros([1, K]).flatten().astype(np.float32),
        )
        gemm_output_variable_name = scope.get_unique_variable_name("gemm_output")
        gemm_attrs = {
            "name": scope.get_unique_operator_name("GeMM"),
            "alpha": 1.0,
            "transB": 1,
        }
        container.add_node(
            op_type="Gemm",
            inputs=[
                operator.inputs[0].full_name,
                centers_variable_name,
                zeros_variable_name,
            ],
            outputs=[gemm_output_variable_name],
            op_version=7,
            **gemm_attrs,
        )

        # Cosine similarity = gemm_output / input_row_norm2 / centers_row_norm2: [N x K]
        div_output_variable_name = scope.get_unique_variable_name("div_output")
        container.add_node(
            op_type="Div",
            inputs=[gemm_output_variable_name, input_row_norm2_variable_name],
            outputs=[div_output_variable_name],
            op_version=7,
        )
        cosine_similarity_output_variable_name = scope.get_unique_variable_name(
            "cosine_similarity_output"
        )
        container.add_node(
            op_type="Div",
            inputs=[div_output_variable_name, centers_row_norm2_variable_name],
            outputs=[cosine_similarity_output_variable_name],
            op_version=7,
        )

        # Cosine distance - 1 = -Cosine similarity: [N x K]
        container.add_node(
            op_type="Neg",
            inputs=[cosine_similarity_output_variable_name],
            outputs=[distance_output_variable_name],
        )
    else:
        raise ValueError(f"Distance measure {op.getDistanceMeasure()} not supported")

    # ArgMin(distance): [N]
    argmin_attrs = {
        "axis": 1,
        "keepdims": 0,
    }
    container.add_node(
        op_type="ArgMin",
        inputs=[distance_output_variable_name],
        outputs=[operator.outputs[0].full_name],
        **argmin_attrs,
    )


register_converter("pyspark.ml.clustering.KMeansModel", convert_sparkml_k_means_model)


def calculate_k_means_model_output_shapes(operator: Operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)
    check_input_and_output_types(operator, good_input_types=[FloatTensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError("Input must be a [N, C]-tensor")

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])


register_shape_calculator(
    "pyspark.ml.clustering.KMeansModel", calculate_k_means_model_output_shapes
)
