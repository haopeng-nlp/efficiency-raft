# -*- coding: utf-8 -*-
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()


with open(here / "requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="raft_baselines",
    version="0.0.1",
    description="RAFT Benchmarks baselines, classifiers, and testing scripts.",
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
)
