# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import onnx
from onnxmltools import convert_h2o


class TestH2OTemporaryFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.converter = importlib.import_module("onnxmltools.convert.h2o.convert")
        self.enterContext(patch.object(tempfile, "tempdir", str(self.root)))
        self.result = onnx.ModelProto()
        self.topology = Mock()
        self.enterContext(
            patch.object(self.converter, "parse_h2o", return_value=self.topology)
        )
        self.enterContext(
            patch.object(self.converter, "convert_topology", return_value=self.result)
        )
        self.paths = []

    def read_mojo(self, path, format):
        self.assertEqual(format, "json")
        self.paths.append(Path(path))
        self.assertEqual(Path(path).read_bytes(), b"model bytes")
        return json.dumps({"params": {"algo": "gbm"}})

    def assert_temporary_files_removed(self):
        self.assertEqual(list(self.root.iterdir()), [])
        for path in self.paths:
            self.assertFalse(path.exists())

    def test_byte_input_is_read_and_removed(self):
        with patch.object(self.converter.h2o, "print_mojo", side_effect=self.read_mojo):
            self.assertIs(convert_h2o(b"model bytes"), self.result)
        self.topology.compile.assert_called_once_with()
        self.assertEqual(len(self.paths), 1)
        self.assert_temporary_files_removed()

    def test_repeated_byte_inputs_do_not_accumulate_files(self):
        with patch.object(self.converter.h2o, "print_mojo", side_effect=self.read_mojo):
            for _ in range(20):
                self.assertIs(convert_h2o(b"model bytes"), self.result)
                self.assert_temporary_files_removed()

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires Linux descriptor counts")
    def test_repeated_failures_do_not_leak_descriptors(self):
        def reject_model(path, format):
            self.read_mojo(path, format)
            raise ValueError("invalid model")

        before = len(list(Path("/proc/self/fd").iterdir()))
        with patch.object(self.converter.h2o, "print_mojo", side_effect=reject_model):
            for _ in range(20):
                with self.assertRaisesRegex(ValueError, "invalid model"):
                    convert_h2o(b"model bytes")
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
        self.assert_temporary_files_removed()

    def test_print_mojo_error_cleans_up(self):
        def reject_model(path, format):
            self.read_mojo(path, format)
            raise ValueError("invalid model")

        with patch.object(self.converter.h2o, "print_mojo", side_effect=reject_model):
            with self.assertRaisesRegex(ValueError, "invalid model"):
                convert_h2o(b"model bytes")
        self.assert_temporary_files_removed()

    def test_invalid_json_cleans_up(self):
        with patch.object(self.converter.h2o, "print_mojo", return_value="not json"):
            with self.assertRaises(json.JSONDecodeError):
                convert_h2o(b"model bytes")
        self.assert_temporary_files_removed()

    def test_unsupported_algorithm_cleans_up(self):
        data = json.dumps({"params": {"algo": "unsupported"}})
        with patch.object(self.converter.h2o, "print_mojo", return_value=data):
            with self.assertRaisesRegex(ValueError, "Model type not supported"):
                convert_h2o(b"model bytes")
        self.assert_temporary_files_removed()

    def test_conversion_error_cleans_up(self):
        self.topology.compile.side_effect = ValueError("cannot compile")
        with patch.object(self.converter.h2o, "print_mojo", side_effect=self.read_mojo):
            with self.assertRaisesRegex(ValueError, "cannot compile"):
                convert_h2o(b"model bytes")
        self.assert_temporary_files_removed()

    def test_invalid_byte_input_cleans_up(self):
        with self.assertRaises(TypeError):
            convert_h2o(object())
        self.assert_temporary_files_removed()

    def test_caller_owned_file_is_preserved_on_success_and_error(self):
        path = self.root / "caller.zip"
        path.write_bytes(b"model bytes")
        with patch.object(self.converter.h2o, "print_mojo", side_effect=self.read_mojo):
            self.assertIs(convert_h2o(str(path)), self.result)
        with patch.object(self.converter.h2o, "print_mojo", side_effect=ValueError("invalid")):
            with self.assertRaisesRegex(ValueError, "invalid"):
                convert_h2o(str(path))
        self.assertEqual(path.read_bytes(), b"model bytes")
        self.assertEqual(list(self.root.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
