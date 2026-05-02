"""
setup.py - Installation Script for Al-Mizan Protocol
============================================================================
Constitutional Autograd for Digital Sovereignty

This file allows the protocol to be installed as a reusable Python package:
    pip install -e .

Or distributed via PyPI:
    pip install al-mizan-protocol

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)
License: Apache 2.0
============================================================================
"""

from setuptools import setup, find_packages
import os

# Read README.md for long description
def read_long_description():
    """Read the long description from README.md."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read version from package (optional, but good practice)
def get_version():
    """Get version from almizan package."""
    with open(os.path.join("almizan", "__init__.py"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"  # fallback

setup(
    # ------------------------------------------------------------------------
    # Package Metadata
    # ------------------------------------------------------------------------
    name="al-mizan-protocol",
    version="1.0.0",
    author="Amine Rekab",
    author_email="amine.rekab@example.com",  # Update with your professional email
    maintainer="Amine Rekab",
    maintainer_email="amine.rekab@example.com",
    
    description="A constitutional autograd engine for digital justice and sovereignty.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # ------------------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------------------
    url="https://github.com/rekabamine91-lgtm/al-mizan-protocol",
    project_urls={
        "Documentation": "https://github.com/rekabamine91-lgtm/al-mizan-protocol#readme",
        "Whitepaper": "https://github.com/rekabamine91-lgtm/al-mizan-protocol/blob/main/paper/main.pdf",
        "Source Code": "https://github.com/rekabamine91-lgtm/al-mizan-protocol",
        "Issue Tracker": "https://github.com/rekabamine91-lgtm/al-mizan-protocol/issues",
        "Changelog": "https://github.com/rekabamine91-lgtm/al-mizan-protocol/releases",
    },
    
    # ------------------------------------------------------------------------
    # Package Configuration
    # ------------------------------------------------------------------------
    packages=find_packages(exclude=["tests", "tests.*", "docs", "paper"]),
    include_package_data=True,
    zip_safe=False,  # Allows access to data files
    
    # ------------------------------------------------------------------------
    # Python Version & Dependencies
    # ------------------------------------------------------------------------
    python_requires=">=3.8, <3.12",  # 3.8 through 3.11 supported
    install_requires=[
        "numpy>=1.21.0,<2.0.0",      # Core numerical operations
        "pandas>=1.3.0,<2.0.0",      # Audit trail data management
        "streamlit>=1.20.0,<2.0.0",  # Interactive dashboard
        "matplotlib>=3.5.0,<4.0.0",  # Static visualizations
        "plotly>=5.17.0,<6.0.0",     # Interactive charts (optional but recommended)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",       # Enhanced statistical visualizations
        ],
        "demo": [
            "imageio>=2.20.0",       # GIF generation for README
        ],
        "all": [
            "pytest>=7.0.0",
            "seaborn>=0.11.0",
            "imageio>=2.20.0",
        ],
    },
    
    # ------------------------------------------------------------------------
    # Console Scripts (Entry Points)
    # ------------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            "almizan-dashboard=app:main",              # Launch dashboard
            "almizan-simulate=simulation:run_simulation",  # Run benchmark
            "almizan-test=pytest:main",                # Run tests (requires pytest)
        ],
    },
    
    # ------------------------------------------------------------------------
    # Classifiers (for PyPI visibility)
    # ------------------------------------------------------------------------
    classifiers=[
        # Maturity
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Ethics :: AI Fairness",
        
        # License
        "License :: OSI Approved :: Apache Software License",
        
        # Programming Languages
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating Systems
        "Operating System :: OS Independent",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # ------------------------------------------------------------------------
    # Keywords for PyPI search
    # ------------------------------------------------------------------------
    keywords=[
        "autograd",
        "fairness",
        "constitutional-ai",
        "algorithmic-justice",
        "digital-sovereignty",
        "deep-learning",
        "zakat",
        "al-qist",
        "ethical-ai",
        "responsible-ai",
    ],
    
    # ------------------------------------------------------------------------
    # Data Files
    # ------------------------------------------------------------------------
    package_data={
        "almizan": ["py.typed"],  # For type hints (optional)
    },
    
    # ------------------------------------------------------------------------
    # Command Line Options
    # ------------------------------------------------------------------------
    options={
        "build": {"build_base": "build"},
        "bdist_wheel": {"universal": False},
    },
    
    # ------------------------------------------------------------------------
    # Additional Metadata
    # ------------------------------------------------------------------------
    license="Apache 2.0",
    platforms=["any"],
    requires_python=">=3.8",
    
    # ------------------------------------------------------------------------
    # Citation
    # ------------------------------------------------------------------------
    # A DOI will be added after Zenodo registration
    # See CITATION.cff for full citation information
)

# ============================================================================
# Installation Instructions:
# ============================================================================
# Development mode (editable):
#     pip install -e .
#
# Install with all extras:
#     pip install -e .[all]
#
# Install with development dependencies:
#     pip install -e .[dev]
#
# Build distribution:
#     python setup.py sdist bdist_wheel
#
# Upload to PyPI (requires twine):
#     twine upload dist/*
# ============================================================================
