#!/usr/bin/env python
# coding: utf-8

from setuptools import setup

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

package_dir = {
    'prpa': 'prpa',
}

setup(
    name='prpa',
    version='1.1.1',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/PostRenderPerspectiveAlign',
    description=u'Post-Render Perspective Align',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir=package_dir,
    packages=[key for key in package_dir],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'torch',
    ],
)
