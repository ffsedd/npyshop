#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
import sys


    
setup(
    name='npyshop',
    version='1.0.0',
    url='https://github.com/ffsedd/npyshop/',
    author='ffsedd',
    author_email='ffsedd@gmail.com',
    description='python image editor',
    packages=['npyshop'],
    #scripts=['qq'],
    install_requires=['send2trash', 'pillow', 'numpy'],
    include_package_data=True,
)
