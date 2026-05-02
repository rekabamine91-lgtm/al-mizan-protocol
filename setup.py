from setuptools import setup, find_packages

setup(
    name="almizan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    author="Rekab Amine",
    description="Al-Mizan Protocol: A socio-technical framework for balanced AI.",
)
