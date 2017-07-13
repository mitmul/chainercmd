#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup_requires = []
install_requires = [
    'chainer==2.0.1',
    'chainercv==0.6.0',
    'pyyaml'
]

setup(
    name='chainerttp',
    version='0.0.1',
    description='Templates for Chainer',
    author='Shunta Saito',
    author_email='shunta.saito@gmail.com',
    url='https://chainertp.readthedocs.io',
    license='MIT License',
    packages=[
        'chainercmd',
    ],
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock', 'nose'],
    entry_points={
        'console_scripts': [
            'chainer=chainercmd.bin.chainercmd:main'
        ]
    }
)
