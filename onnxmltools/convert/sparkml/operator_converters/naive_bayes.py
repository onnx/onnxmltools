# SPDX-License-Identifier: Apache-2.0

import numpy

from ....proto import onnx_proto
from ...common.data_types import FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator


def convert_sparkml_naive_bayes(scope, operator, container):
    op = operator.raw_operator
    model_type = op.getOrDefault("modelType")
    theta_np = numpy.array(op.theta.toArray())
    transposed_theta = numpy.transpose(theta_np)
    raw_predictions_tensor = scope.get_unique_variable_name("raw_predictions_tensor")
    if model_type == "bernoulli":
        negTheta = numpy.log(1.0 - numpy.exp(theta_np))
        thetaMinusLogTheta = numpy.transpose(theta_np - negTheta)
        ones = numpy.ones((theta_np.shape[1], 1))
        negThetaSum = numpy.matmul(negTheta, ones).flatten()
        thetaMinusLogTheta_tensor = scope.get_unique_variable_name(
            "thetaMinusLogTheta_tensor"
        )
        container.add_initializer(
            thetaMinusLogTheta_tensor,
            onnx_proto.TensorProto.FLOAT,
            list(thetaMinusLogTheta.shape),
            thetaMinusLogTheta.flatten().tolist(),
        )
        negThetaSum_tensor = scope.get_unique_variable_name("negThetaSum_tensor")
        container.add_initializer(
            negThetaSum_tensor,
            onnx_proto.TensorProto.FLOAT,
            list(negThetaSum.shape),
            negThetaSum.flatten().tolist(),
        )
        prior_tensor = scope.get_unique_variable_name("prior_tensor")
        container.add_initializer(
            prior_tensor,
            onnx_proto.TensorProto.FLOAT,
            [len(op.pi)],
            op.pi.flatten().tolist(),
        )
        probability1_output = scope.get_unique_variable_name("temp_probability")
        container.add_node(
            "MatMul",
            [operator.input_full_names[0], thetaMinusLogTheta_tensor],
            probability1_output,
            op_domain="ai.onnx",
            name=scope.get_unique_operator_name("MatMul"),
            op_version=9,
        )
        probability2_output = scope.get_unique_variable_name("temp_probability")
        container.add_node(
            "Add",
            [probability1_output, prior_tensor],
            probability2_output,
            op_domain="ai.onnx",
            name=scope.get_unique_operator_name("Add"),
            op_version=7,
        )
        container.add_node(
            "Add",
            [probability2_output, negThetaSum_tensor],
            raw_predictions_tensor,
            op_domain="ai.onnx",
            name=scope.get_unique_operator_name("Add"),
            op_version=7,
        )
    else:
        probability1_output = scope.get_unique_variable_name("temp_probability")
        theta_tensor = scope.get_unique_variable_name("theta_tensor")
        container.add_initializer(
            theta_tensor,
            onnx_proto.TensorProto.FLOAT,
            list(transposed_theta.shape),
            transposed_theta.flatten().tolist(),
        )
        container.add_node(
            "MatMul",
            [operator.input_full_names[0], theta_tensor],
            probability1_output,
            op_domain="ai.onnx",
            name=scope.get_unique_operator_name("MatMul"),
            op_version=1,
        )
        prior_tensor = scope.get_unique_variable_name("raw_predictions_tensor")
        container.add_initializer(
            prior_tensor, onnx_proto.TensorProto.FLOAT, [len(op.pi)], op.pi
        )
        container.add_node(
            "Add",
            [probability1_output, prior_tensor],
            raw_predictions_tensor,
            op_domain="ai.onnx",
            name=scope.get_unique_operator_name("Add"),
            op_version=7,
        )
    argmax_tensor = scope.get_unique_variable_name("argmax_tensor")
    container.add_node(
        "ArgMax",
        raw_predictions_tensor,
        argmax_tensor,
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("ArgMax"),
        op_version=1,
        axis=1,
    )
    container.add_node(
        "Cast",
        argmax_tensor,
        operator.output_full_names[0],
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("Cast"),
        op_version=9,
        to=1,
    )
    # Now we need to calculate Probabilities from rawPredictions
    # print('prediction:', numpy.argmax(result, 1))
    # max_log = numpy.max(result, 1).reshape(result.shape[0], 1)
    max_prediction_tensor = scope.get_unique_variable_name("max_prediction_tensor")
    container.add_node(
        "ReduceMax",
        raw_predictions_tensor,
        max_prediction_tensor,
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("ReduceMax"),
        op_version=1,
        axes=[1],
        keepdims=1,
    )
    # sub_result = result - max_log
    raw_minus_max_tensor = scope.get_unique_variable_name("raw_minus_max_tensor")
    container.add_node(
        "Sub",
        [raw_predictions_tensor, max_prediction_tensor],
        raw_minus_max_tensor,
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("Sub"),
        op_version=7,
    )
    # exp_result = numpy.exp(sub_result)
    exp_tensor = scope.get_unique_variable_name("exp_tensor")
    container.add_node(
        "Exp",
        raw_minus_max_tensor,
        exp_tensor,
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("Exp"),
        op_version=6,
    )
    # sum_log = numpy.sum(exp_result, 1).reshape(result.shape[0], 1)
    sum_prediction_tensor = scope.get_unique_variable_name("sum_prediction_tensor")
    container.add_node(
        "ReduceSum",
        exp_tensor,
        sum_prediction_tensor,
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("ReduceSum"),
        op_version=1,
        axes=[1],
        keepdims=1,
    )
    # probabilities = exp_result / sum_log
    container.add_node(
        "Div",
        [exp_tensor, sum_prediction_tensor],
        operator.output_full_names[1],
        op_domain="ai.onnx",
        name=scope.get_unique_operator_name("Div"),
        op_version=7,
    )


register_converter(
    "pyspark.ml.classification.NaiveBayesModel", convert_sparkml_naive_bayes
)


def calculate_sparkml_naive_bayes_output_shapes(operator):
    check_input_and_output_numbers(operator, output_count_range=2)
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType],
        good_output_types=[FloatTensorType, FloatTensorType],
    )
    N = operator.inputs[0].type.shape[0]
    C = operator.raw_operator.numClasses
    operator.outputs[0].type = FloatTensorType([N, 1])
    operator.outputs[1].type = FloatTensorType([N, C])


register_shape_calculator(
    "pyspark.ml.classification.NaiveBayesModel",
    calculate_sparkml_naive_bayes_output_shapes,
)
