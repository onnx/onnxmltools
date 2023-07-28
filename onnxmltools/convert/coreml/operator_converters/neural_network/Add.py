# SPDX-License-Identifier: Apache-2.0

from .....proto import onnx_proto
from ....common._registration import register_converter
from ....common._apply_operation import apply_add


def convert_add(scope, operator, container):
    if len(operator.input_full_names) == 1:
        scaler_name = scope.get_unique_variable_name(operator.full_name + "_B")
        container.add_initializer(
            scaler_name,
            onnx_proto.TensorProto.FLOAT,
            [],
            [operator.raw_operator.add.alpha],
        )
        inputs = [operator.inputs[0].full_name, scaler_name]
        broadcast = 1
        apply_add(
            scope,
            inputs,
            operator.output_full_names,
            container,
            operator_name=operator.full_name,
            broadcast=broadcast,
        )
    else:
        inputs = operator.input_full_names

        if len(inputs) == 2:
            left_tensor = operator.inputs[0].full_name
            right_tensor = operator.inputs[1].full_name
            if operator.inputs[0].type.shape != operator.inputs[1].type.shape:
                broadcast = 1
            else:
                broadcast = 0

            apply_add(
                scope,
                [left_tensor, right_tensor],
                operator.outputs[0].full_name,
                container,
                operator_name=operator.full_name,
                broadcast=broadcast,
            )
        else:
            left_tensor = operator.inputs[0].full_name
            right_tensor = operator.inputs[1].full_name

            # Sum up the first two inputs
            intermediate_tensor_name = scope.get_unique_variable_name("buffer_tensor")
            apply_add(
                scope,
                [left_tensor, right_tensor],
                intermediate_tensor_name,
                container,
                operator_name=operator.full_name,
                broadcast=1,
            )

            # Accumulate other inputs onto intermediate tensors.
            # Note that we may use the original operator's output as
            # the last intermediate tensor.
            for i in range(2, len(inputs)):
                left_tensor = intermediate_tensor_name
                right_tensor = inputs[i].full_name
                if i != len(inputs) - 1:
                    intermediate_tensor_name = scope.get_unique_variable_name(
                        "buffer_tensor"
                    )
                else:
                    intermediate_tensor_name = operator.outputs[0].full_name
                apply_add(
                    scope,
                    [left_tensor, right_tensor],
                    intermediate_tensor_name,
                    container,
                    broadcast=1,
                )


register_converter("add", convert_add)
