from setuptools import setup, find_packages

setup(
    name="almizan-protocol",
    version="1.0.0",
    author="Amine Rekab",
    description="A foundational framework for distributive justice in neural networks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rekabamine91-lgtm/al-mizan-protocol",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "streamlit>=1.0.0",
        "pandas>=1.3.0",
        "pytest>=6.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'almizan-dashboard=app:main',
        ],
    },
)
