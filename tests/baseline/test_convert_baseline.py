# SPDX-License-Identifier: Apache-2.0

"""
Tests converters for a baseline.
"""
import os
import re
import unittest
from onnxmltools.convert import convert_coreml
import coremltools


class TestBaseLine(unittest.TestCase):

    def check_baseline(self, input_file, ref_file):
        diff = self.get_diff(input_file, ref_file)
        return self.normalize_diff(diff)

    def get_diff(self, input_file, ref_file):
        this = os.path.dirname(__file__)
        coreml_file = os.path.join(this, "models", input_file)
        cml = coremltools.utils.load_spec(coreml_file)
        onnx_model = convert_coreml(cml)
        output_dir = os.path.join(this, "outmodels")
        output_file = os.path.join(this, "outmodels", ref_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w') as f:
            f.write(str(onnx_model))
        reference_model = os.path.join(this, "models", ref_file)
        with open(reference_model, 'r') as ref_file:
            with open(output_file, 'r') as output_file:
                diff = set(ref_file).difference(output_file)
        return diff

    def normalize_diff(self, diff):
        invalid_comparisons = []
        invalid_comparisons.append(re.compile('producer_version: \"\d+\.\d+\.\d+\.\d+.*'))
        invalid_comparisons.append(re.compile('\s+name: \".*'))
        invalid_comparisons.append(re.compile('ir_version: \d+'))
        invalid_comparisons.append(re.compile('\s+'))
        valid_diff = set()
        for line in diff:
            if any(comparison.match(line) for comparison in invalid_comparisons):
                continue
            valid_diff.add(line)
        return valid_diff

    def test_keras2coreml_Dense_ImageNet_small(self):
        """
        Converting keras2coreml_Dense_ImageNet_small using onnxmltools and comparing with last known good result
        """
        self.assertFalse(self.check_baseline(
            "keras2coreml_Dense_ImageNet_small.mlmodel", "keras2coreml_Dense_ImageNet_small.json"))


if __name__ == "__main__":
    unittest.main()
