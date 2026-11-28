# SPDX-License-Identifier: Apache-2.0

"""Validate release distributions and smoke-test their clean installations."""

import argparse
from email.parser import BytesParser
from importlib.metadata import version
from pathlib import Path
import tarfile
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def metadata_version(data):
    metadata = BytesParser().parsebytes(data)
    if canonicalize_name(metadata.get("Name", "")) != "onnxmltools":
        raise ValueError("Distribution metadata must name onnxmltools")
    return Version(metadata.get("Version", ""))


def validate_artifacts(directory, tag=None):
    files = [path for path in Path(directory).iterdir() if path.is_file()]
    wheels = [path for path in files if path.name.endswith(".whl")]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")

    wheel, sdist = wheels[0], sdists[0]
    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel.name)
    sdist_name, sdist_version = parse_sdist_filename(sdist.name)
    if wheel_name != "onnxmltools" or sdist_name != "onnxmltools":
        raise ValueError("Distribution filenames must name onnxmltools")

    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected one wheel METADATA file")
        wheel_metadata_version = metadata_version(archive.read(metadata_files[0]))
    with tarfile.open(sdist) as archive:
        # Read the archive's top-level package metadata without extracting it.
        metadata_file = archive.extractfile(f"{sdist.name[:-7]}/PKG-INFO")
        if metadata_file is None:
            raise ValueError("Missing source distribution PKG-INFO")
        sdist_metadata_version = metadata_version(metadata_file.read())

    versions = {wheel_version, sdist_version, wheel_metadata_version, sdist_metadata_version}
    if len(versions) != 1:
        raise ValueError(
            "Wheel and source distribution filenames/metadata disagree on version"
        )
    if tag is not None and Version(tag) != wheel_version:
        raise ValueError(f"Release tag {tag!r} does not match package version {wheel_version}")
    return wheel_version


def validate_installed_versions(expected, installed, imported):
    if Version(installed) != Version(expected) or Version(imported) != Version(expected):
        raise ValueError(
            f"Version mismatch: expected {expected}, installed metadata {installed}, "
            f"onnxmltools.__version__ {imported}"
        )


def check_installed(expected, source_root):
    import onnxmltools

    source_root = Path(source_root).resolve()
    if Path.cwd().resolve().is_relative_to(source_root):
        raise ValueError("Run installation checks outside the source checkout")
    if Path(onnxmltools.__file__).resolve().is_relative_to(source_root):
        raise ValueError("Imported onnxmltools from the source checkout")
    validate_installed_versions(expected, version("onnxmltools"), onnxmltools.__version__)
    smoke_test_conversions()
    print(f"Validated installed onnxmltools {expected} from {onnxmltools.__file__}")


def smoke_test_conversions():
    import numpy as np
    import onnx
    import onnxruntime as ort
    from lightgbm import LGBMClassifier
    from xgboost import XGBRegressor

    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    values = np.random.RandomState(0).normal(size=(30, 3)).astype(np.float32)
    initial_types = [("X", FloatTensorType([None, 3]))]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def run(model):
        onnx.checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        return session.run(None, {"X": values})

    regressor = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, random_state=0)
    regressor.fit(values, values[:, 0] + values[:, 1])
    regression = onnxmltools.convert_xgboost(regressor, initial_types=initial_types)
    np.testing.assert_allclose(
        run(regression)[0].ravel(), regressor.predict(values), rtol=1e-5, atol=1e-6
    )

    classifier = LGBMClassifier(
        n_estimators=3,
        num_leaves=4,
        min_child_samples=1,
        n_jobs=1,
        random_state=0,
        verbosity=-1,
    )
    classifier.fit(values, (values[:, 0] > 0).astype(np.int64))
    classification = onnxmltools.convert_lightgbm(
        classifier, initial_types=initial_types, zipmap=False
    )
    labels, probabilities = run(classification)
    np.testing.assert_array_equal(labels.ravel(), classifier.predict(values))
    np.testing.assert_allclose(
        probabilities, classifier.predict_proba(values), rtol=1e-5, atol=1e-6
    )
    print("XGBoost regression and LightGBM classification smoke tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("directory", type=Path)
    artifacts.add_argument("--tag")
    artifacts.add_argument("--github-output", type=Path)
    installed = commands.add_parser("installed")
    installed.add_argument("--version", required=True)
    installed.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "artifacts":
        expected = validate_artifacts(args.directory, args.tag)
        print(f"Validated wheel and source distribution version {expected}")
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"version={expected}\n")
    else:
        check_installed(args.version, args.source_root)


if __name__ == "__main__":
    main()
