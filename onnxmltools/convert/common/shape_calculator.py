# SPDX-License-Identifier: Apache-2.0

from .data_types import FloatTensorType


def check_input_and_output_numbers(
    operator, input_count_range=None, output_count_range=None
):
    """
    Check if the number of input(s)/output(s) is correct

    :param operator: A Operator object
    :param input_count_range: A list of two integers or an integer. If it's a list the first/second element is the
    minimal/maximal number of inputs. If it's an integer, it is equivalent to specify that number twice in a list. For
    infinite ranges like 5 to infinity, you need to use [5, None].
    :param output_count_range: A list of two integers or an integer. See input_count_range for its format.
    """
    if isinstance(input_count_range, list):
        min_input_count = input_count_range[0]
        max_input_count = input_count_range[1]
    elif isinstance(input_count_range, int) or input_count_range is None:
        min_input_count = input_count_range
        max_input_count = input_count_range
    else:
        raise RuntimeError("input_count_range must be a list or an integer")

    if isinstance(output_count_range, list):
        min_output_count = output_count_range[0]
        max_output_count = output_count_range[1]
    elif isinstance(output_count_range, int) or output_count_range is None:
        min_output_count = output_count_range
        max_output_count = output_count_range
    else:
        raise RuntimeError("output_count_range must be a list or an integer")

    if min_input_count is not None and len(operator.inputs) < min_input_count:
        raise RuntimeError(
            "For operator %s (type: %s), at least %s input(s) is(are) required but we got %s input(s) which are %s"
            % (
                operator.full_name,
                operator.type,
                min_input_count,
                len(operator.inputs),
                operator.input_full_names,
            )
        )

    if max_input_count is not None and len(operator.inputs) > max_input_count:
        raise RuntimeError(
            "For operator %s (type: %s), at most %s input(s) is(are) supported but we got %s input(s) which are %s"
            % (
                operator.full_name,
                operator.type,
                max_input_count,
                len(operator.inputs),
                operator.input_full_names,
            )
        )

    if min_output_count is not None and len(operator.outputs) < min_output_count:
        raise RuntimeError(
            "For operator %s (type: %s), at least %s output(s) is(are) produced but we got %s output(s) which are %s"
            % (
                operator.full_name,
                operator.type,
                min_output_count,
                len(operator.outputs),
                operator.output_full_names,
            )
        )

    if max_output_count is not None and len(operator.outputs) > max_output_count:
        raise RuntimeError(
            "For operator %s (type: %s), at most %s outputs(s) is(are) supported but we got %s output(s) which are %s"
            % (
                operator.full_name,
                operator.type,
                max_output_count,
                len(operator.outputs),
                operator.output_full_names,
            )
        )


def check_input_and_output_types(
    operator, good_input_types=None, good_output_types=None
):
    """
    Check if the type(s) of input(s)/output(s) is(are) correct

    :param operator: A Operator object
    :param good_input_types: A list of allowed input types (e.g., [FloatTensorType, Int64TensorType]) or None. None
    means that we skip the check of the input types.
    :param good_output_types: A list of allowed output types. See good_input_types for its format.
    """
    if good_input_types is not None:
        for variable in operator.inputs:
            if type(variable.type) not in good_input_types:
                raise RuntimeError(
                    "Operator %s (type: %s) got an input %s with a wrong type %s. Only %s are allowed"
                    % (
                        operator.full_name,
                        operator.type,
                        variable.full_name,
                        type(variable.type),
                        good_input_types,
                    )
                )

    if good_output_types is not None:
        for variable in operator.outputs:
            if type(variable.type) not in good_output_types:
                raise RuntimeError(
                    "Operator %s (type: %s) got an output %s with a wrong type %s. Only %s are allowed"
                    % (
                        operator.full_name,
                        operator.type,
                        variable.full_name,
                        type(variable.type),
                        good_output_types,
                    )
                )


def calculate_linear_regressor_output_shapes(operator):
    """
    Allowed input/output patterns are
        1. [N, C] ---> [N, 1]

    This operator produces a scalar prediction for every example in a batch. If the input batch size is N, the output
    shape may be [N, 1].
    """
    check_input_and_output_numbers(operator, input_count_range=1, output_count_range=1)

    N = operator.inputs[0].type.shape[0]
    op = operator.raw_operator
    if hasattr(op, "n_outputs_"):
        nout = op.n_outputs_
    else:
        nout = 1
    operator.outputs[0].type = FloatTensorType([N, nout])
