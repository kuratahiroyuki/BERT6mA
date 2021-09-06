#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:36:42 2020

@author: tsukiyamashou
"""
import sys
from setuptools import setup, find_packages
import os
from setuptools import find_packages
import setuptools
from setuptools.command.install_scripts import install_scripts
from setuptools.command.install import install
from distutils import log

if sys.version_info.major < 3:
    sys.exit('Sorry, sierra-local requires Python 3.x')

def _requires_from_file(filename):
    print(os.getcwd())
    return open(filename).read().splitlines()

if __name__ == "__main__":
                
    setup(
        name='BERT6mA',
        version='0.0.1',
        description="This package called BERT6mA is used to predict 6mA sites",
        author="Kyushu institute of technology. Kurata laboratory.",
        install_requires = _requires_from_file('requirements.txt'),
        packages = ["predict", "deep_model", "w2v_model"],
        package_data = {
            'deep_model': ['6mA_A.thaliana/deep_model', '6mA_C.elegans/deep_model', '6mA_C.equisetifolia/deep_model', '6mA_D.melanogaster/deep_model', '6mA_F.vesca/deep_model', '6mA_H.sapiens/deep_model', '6mA_R.chinensis/deep_model', '6mA_S.cerevisiae/deep_model', '6mA_T.thermophile/deep_model', '6mA_Ts.SUP5-1/deep_model', '6mA_Xoc.BLS256/deep_model', '6mA_R.chinensis++/deep_model'],
            'w2v_model': ['dna_w2v_100.pt'],
        },
        entry_points={
            'console_scripts':[
                'bert6mA = predict.testing_w2v:main',
            ]
        },
        classifiers=[
        "Programming Language :: Python :: 3.8.0",
        "License :: Apache License 2.0"
        ],
    )































