from setuptools import setup


setup(
    name="opdxread",
    version="0.1.0",
    description="Lib for reading Vision64 profilometry data from .OPDx files.",
    packages=["opdxread"],
    license="LGPL",
    author="djdt",
    install_requires=[
        "numpy",
    ]
)
