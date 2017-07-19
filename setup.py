#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup_requires = []
install_requires = [
    'chainer<3',
    'pyyaml>=3.12'
]

setup(
    name='chainercmd',
    version='1.0.0a5',
    description='Command Line Tools for Chainer',
    author='Shunta Saito',
    author_email='shunta.saito@gmail.com',
    url='https://github.com/mitmul/chainercmd',
    license='MIT',
    packages=[
        'chainercmd',
        'chainercmd.bin',
        'chainercmd.config',
        'chainercmd.template',
    ],
    package_data={
        'chainercmd.template': ['config.yml']
    },
    keywords='chainer deeplearning',
    python_requires='>=3.4.4',
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
