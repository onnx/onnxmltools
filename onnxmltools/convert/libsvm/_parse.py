# SPDX-License-Identifier: Apache-2.0

from onnxconverter_common.data_types import FloatTensorType
from ..common._container import LibSvmModelContainer
from ..common._topology import Topology


def _parse_libsvm_simple_model(scope, model, inputs):
    """
    This function handles all non-pipeline models.

    :param scope: Scope object
    :param model: A libsvm object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: A list of output variables which will be passed to next stage
    """

    if model.get_svm_type() in (0, 1):
        label_variable = scope.declare_local_variable("label", FloatTensorType())
        probability_map_variable = scope.declare_local_variable(
            "probabilities", FloatTensorType()
        )
        this_operator = scope.declare_local_operator("LibSvmSVC", model)
        this_operator.inputs = inputs
        this_operator.outputs.append(label_variable)
        this_operator.outputs.append(probability_map_variable)
    elif model.get_svm_type() in (4, 3):
        # We assume that all scikit-learn operator can only produce a single float tensor.
        variable = scope.declare_local_variable("variable", FloatTensorType())
        this_operator = scope.declare_local_operator("LibSvmSVR", model)
        this_operator.inputs = inputs
        this_operator.outputs.append(variable)
    else:
        raise ValueError("Unknown SVM type '{0}'".format(model.get_svm_type()))
    return this_operator.outputs


def _parse_libsvm(scope, model, inputs):
    """
    This is a delegate function. It doesn't nothing but invoke
    the correct parsing function according to the input
    model's type.

    :param scope: Scope object
    :param model: A scikit-learn object (e.g., OneHotEncoder and LogisticRegression)
    :param inputs: A list of variables
    :return: The output variables produced by the input model
    """
    return _parse_libsvm_simple_model(scope, model, inputs)


def parse_libsvm(
    model,
    initial_types=None,
    target_opset=None,
    custom_conversion_functions=None,
    custom_shape_calculators=None,
):
    # Put svmlib object into an abstract container so that our framework
    # can work seamlessly on models created
    # with different machine learning tools.
    raw_model_container = LibSvmModelContainer(model)

    # Declare a computational graph. It will become a representation of
    # the input scikit-learn model after parsing.
    topology = Topology(
        raw_model_container,
        default_batch_size="None",
        initial_types=initial_types,
        target_opset=target_opset,
        custom_conversion_functions=custom_conversion_functions,
        custom_shape_calculators=custom_shape_calculators,
    )

    # Declare an object to provide variables' and operators' naming mechanism.
    # In contrast to CoreML, one global scope
    # is enough for parsing scikit-learn models.
    scope = topology.declare_scope("__root__")

    # Declare input variables. They should be the inputs of the scikit-learn model
    # you want to convert into ONNX.
    inputs = []
    for var_name, initial_type in initial_types:
        inputs.append(scope.declare_local_variable(var_name, initial_type))

    # The object raw_model_container is a part of the topology we're going to return.
    # We use it to store the inputs of
    # the libsvm's computational graph.
    for variable in inputs:
        raw_model_container.add_input(variable)

    # Parse the input libsvm model as a Topology object.
    outputs = _parse_libsvm(scope, model, inputs)

    # THe object raw_model_container is a part of the topology we're going to return.
    # We use it to store the outputs of
    # the scikit-learn's computational graph.
    for variable in outputs:
        raw_model_container.add_output(variable)

    return topology
