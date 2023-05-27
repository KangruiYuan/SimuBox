#!python
# -*- coding:utf-8 -*-
from __future__ import print_function
from setuptools import setup, find_packages
import SimuBox

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="SimuBox",
    version=SimuBox.__version__,
    author="Alkaid Yuan",
    author_email="kryuan@qq.com",
    description="free python package to do some science calculation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="",
    packages=find_packages(),
    install_requires=[
        "pandas >= 1.4",
        "scipy >= 1.10.0",
        "matplotlib >= 3.6.2",
        "numpy >= 1.19.5",
        "opencv-contrib-python >= 4.7.0.72",
        "opencv-python >= 4.5.3.56",
        "opencv-python-headless >= 3.4.18.65"
    ],
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
