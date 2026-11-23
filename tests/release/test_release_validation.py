# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from packaging.version import Version


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class TestReleaseValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifacts(
        self,
        wheel_version="1.17.0",
        sdist_version="1.17.0",
        wheel_metadata="1.17.0",
        sdist_metadata="1.17.0",
        name="onnxmltools",
    ):
        def metadata(version):
            return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n".encode()

        with zipfile.ZipFile(
            self.root / f"onnxmltools-{wheel_version}-py3-none-any.whl", "w"
        ) as wheel:
            wheel.writestr(
                f"onnxmltools-{wheel_version}.dist-info/METADATA", metadata(wheel_metadata)
            )
        with tarfile.open(self.root / f"onnxmltools-{sdist_version}.tar.gz", "w:gz") as sdist:
            data = metadata(sdist_metadata)
            member = tarfile.TarInfo(f"onnxmltools-{sdist_version}/PKG-INFO")
            member.size = len(data)
            sdist.addfile(member, io.BytesIO(data))

    def test_matching_release_and_nonrelease_builds(self):
        self.artifacts()
        for tag in (None, "1.17.0", "v1.17.0"):
            with self.subTest(tag=tag):
                self.assertEqual(release.validate_artifacts(self.root, tag), Version("1.17.0"))

    def test_matching_release_candidate(self):
        self.artifacts("1.17.0rc1", "1.17.0rc1", "1.17.0rc1", "1.17.0rc1")
        self.assertEqual(
            release.validate_artifacts(self.root, "v1.17.0rc1"), Version("1.17.0rc1")
        )

    def test_wrong_release_tag_is_rejected(self):
        self.artifacts()
        with self.assertRaisesRegex(ValueError, "Release tag.*does not match"):
            release.validate_artifacts(self.root, "v1.18.0")

    def test_empty_or_invalid_release_tag_is_rejected(self):
        self.artifacts()
        for tag in ("", "main"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.validate_artifacts(self.root, tag)

    def test_different_distribution_versions_are_rejected(self):
        self.artifacts(sdist_version="1.18.0", sdist_metadata="1.18.0")
        with self.assertRaisesRegex(ValueError, "disagree on version"):
            release.validate_artifacts(self.root)

    def test_wrong_wheel_metadata_version_is_rejected(self):
        self.artifacts(wheel_metadata="1.18.0")
        with self.assertRaisesRegex(ValueError, "disagree on version"):
            release.validate_artifacts(self.root)

    def test_wrong_sdist_metadata_version_is_rejected(self):
        self.artifacts(sdist_metadata="1.18.0")
        with self.assertRaisesRegex(ValueError, "disagree on version"):
            release.validate_artifacts(self.root)

    def test_wrong_package_name_is_rejected(self):
        self.artifacts(name="another-package")
        with self.assertRaisesRegex(ValueError, "must name onnxmltools"):
            release.validate_artifacts(self.root)

    def test_missing_artifacts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one wheel and one source"):
            release.validate_artifacts(self.root)

    def test_extra_artifact_is_rejected(self):
        self.artifacts()
        (self.root / "unexpected.whl").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "exactly one wheel and one source"):
            release.validate_artifacts(self.root)

    def test_installed_and_imported_versions_must_match(self):
        release.validate_installed_versions("1.17.0", "1.17.0", "1.17.0")
        for installed, imported in (("1.16.0", "1.17.0"), ("1.17.0", "1.16.0")):
            with self.subTest(installed=installed, imported=imported):
                with self.assertRaisesRegex(ValueError, "Version mismatch"):
                    release.validate_installed_versions("1.17.0", installed, imported)

    def test_importing_source_checkout_is_rejected(self):
        package = SimpleNamespace(__file__=str(self.root / "onnxmltools/__init__.py"))
        with patch.dict(sys.modules, {"onnxmltools": package}):
            with self.assertRaisesRegex(ValueError, "Imported onnxmltools from the source"):
                release.check_installed("1.17.0", self.root)

    def test_running_in_source_checkout_is_rejected(self):
        package = SimpleNamespace(__file__=str(self.root / "onnxmltools/__init__.py"))
        with patch.dict(sys.modules, {"onnxmltools": package}):
            with self.assertRaisesRegex(ValueError, "outside the source checkout"):
                release.check_installed("1.17.0", Path.cwd())


if __name__ == "__main__":
    unittest.main()
