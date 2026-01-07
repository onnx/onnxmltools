# SPDX-License-Identifier: Apache-2.0

from distutils.core import setup
from setuptools import find_packages
import os

this = os.path.dirname(__file__)

with open(os.path.join(this, "requirements.txt"), "r") as f:
    requirements = [
        _ for _ in [_.strip("\r\n ") for _ in f.readlines()] if _ is not None
    ]

packages = find_packages()
assert packages

# read version from the package file.
version_str = "1.0.0"
with open(os.path.join(this, "onnxmltools/__init__.py"), "r") as f:
    line = [
        _
        for _ in [_.strip("\r\n ") for _ in f.readlines()]
        if _.startswith("__version__")
    ]
    if len(line) > 0:
        version_str = line[0].split("=")[1].strip('" ')

README = os.path.join(os.getcwd(), "README.md")
with open(README) as f:
    long_description = f.read()

