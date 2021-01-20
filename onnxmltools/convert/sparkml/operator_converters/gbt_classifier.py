# SPDX-License-Identifier: Apache-2.0

from onnx import onnx_pb as onnx_proto
from pyspark.ml.classification import GBTClassificationModel

from ...common._apply_operation import apply_neg, apply_concat, apply_mul, apply_exp, apply_add, \
    apply_argmax, apply_matmul
from ...common.data_types import Int64TensorType, FloatTensorType
from ...common.utils import check_input_and_output_numbers, check_input_and_output_types
from ...common._registration import register_converter, register_shape_calculator
from ..ops_names import get_sparkml_operator_name
from .decision_tree_regressor import convert_decision_tree_regressor


def convert_gbt_classifier(scope, operator, container):
    op = operator.raw_operator
    regressor_output_names = []
    # spark implementation uses DecisionTreeRegressor (and not Classifier) for each tree in this forest
    for tree_model in op.trees:
        regressor_op = scope.declare_local_operator(get_sparkml_operator_name(type(tree_model)), tree_model)
        regressor_op.raw_params = operator.raw_params
        regressor_op.inputs = operator.inputs
        regressor_output = scope.declare_local_variable('regressor_prediction', FloatTensorType())
        regressor_output_names.append(regressor_output.full_name)
        regressor_op.outputs.append(regressor_output)
        convert_decision_tree_regressor(scope, regressor_op, container)
        regressor_op.is_evaluated = True

    targets_tensor = scope.get_unique_variable_name('target_tensor')
    weights_tensor = scope.get_unique_variable_name('weights_tensor')
    container.add_initializer(weights_tensor, onnx_proto.TensorProto.FLOAT, [len(op.treeWeights), 1], op.treeWeights)
    concatenated_predictions = scope.get_unique_variable_name('concatenated_predictions_tensor')
    apply_concat(scope, regressor_output_names, concatenated_predictions, container, axis=1)
    apply_matmul(scope, [concatenated_predictions, weights_tensor], targets_tensor, container)

    # this is to calculate prediction and probability given the raw_prediction (= [-target, target])
    targets_neg_tensor = scope.get_unique_variable_name('target_neg_tensor')
    apply_neg(scope, targets_tensor, targets_neg_tensor, container)
    raw_prediction_tensor = scope.get_unique_variable_name('raw_prediction_tensor')
    apply_concat(scope, [targets_neg_tensor, targets_tensor], raw_prediction_tensor, container,
                 axis=1)
    if isinstance(op, GBTClassificationModel):
        # this section is only for the classifier; for the regressor we don't calculate the probability
        minus_two = scope.get_unique_variable_name('minus_two_tensor')
        container.add_initializer(minus_two, onnx_proto.TensorProto.FLOAT, [1], [-2.0])
        mul_output_tensor = scope.get_unique_variable_name('mul_output_tensor')
        apply_mul(scope, [raw_prediction_tensor, minus_two], mul_output_tensor, container)
        exp_output_tensor = scope.get_unique_variable_name('exp_output_tensor')
        apply_exp(scope, mul_output_tensor, exp_output_tensor, container)
        one_tensor = scope.get_unique_variable_name('one_tensor')
        container.add_initializer(one_tensor, onnx_proto.TensorProto.FLOAT, [1], [1.0])
        add_output_tensor = scope.get_unique_variable_name('add_output_tensor')
        apply_add(scope, [exp_output_tensor, one_tensor], add_output_tensor, container)
        container.add_node('Reciprocal', add_output_tensor, operator.outputs[1].full_name,
                           name=scope.get_unique_operator_name('Reciprocal'),
                           op_version=6)
    # to get Prediction from rawPrediction (or probability)
    apply_argmax(scope, raw_prediction_tensor, operator.outputs[0].full_name, container,
                 axis=1, keepdims=0)


register_converter('pyspark.ml.classification.GBTClassificationModel', convert_gbt_classifier)
register_converter('pyspark.ml.regression.GBTRegressionModel', convert_gbt_classifier)


def calculate_gbt_classifier_output_shapes(operator):
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=[1, 2])
    check_input_and_output_types(operator, good_input_types=[FloatTensorType, Int64TensorType])
    if len(operator.inputs[0].type.shape) != 2:
        raise RuntimeError('Input must be a [N, C]-tensor')

    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])
    if isinstance(operator.raw_operator, GBTClassificationModel):
        class_count = 2
        operator.outputs[1].type = FloatTensorType([N, class_count])


register_shape_calculator('pyspark.ml.classification.GBTClassificationModel',
                          calculate_gbt_classifier_output_shapes)
register_shape_calculator('pyspark.ml.regression.GBTRegressionModel',
                          calculate_gbt_classifier_output_shapes)
