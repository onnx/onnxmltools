# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

from onnxmltools.convert.common._container import Operator


class TestOperatorState(unittest.TestCase):
    @staticmethod
    def operator(name):
        return Operator(name, "scope", "type", object(), 18)

    def test_state_is_cached_per_operator(self):
        first_operator = self.operator("first")
        second_operator = self.operator("second")
        first_value = object()
        second_value = object()
        first_factory = Mock(return_value=first_value)
        second_factory = Mock(return_value=second_value)

        self.assertIs(
            first_operator.get_or_create_local_state("model-parameters", first_factory),
            first_value,
        )
        self.assertIs(
            first_operator.get_or_create_local_state("model-parameters", first_factory),
            first_value,
        )
        self.assertIs(
            second_operator.get_or_create_local_state("model-parameters", second_factory),
            second_value,
        )

        first_factory.assert_called_once_with()
        second_factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
