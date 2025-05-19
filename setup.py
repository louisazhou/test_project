#!/usr/bin/env python
"""
Setup script for the RCA package.
"""

from setuptools import setup, find_packages

# Core dependencies required for the package
requirements = [
    "pandas>=1.0.0",
    "numpy>=1.18.0",
    "matplotlib>=3.1.0",
    "seaborn>=0.11.0",
    "scipy>=1.4.0",
    "jinja2>=2.11.0",
    "pyyaml>=5.1.0",
    "python-pptx>=0.6.18",
]

# Optional dependencies for Google API integration
google_api_deps = [
    "google-api-python-client>=2.0.0",
    "google-auth-httplib2>=0.1.0",
    "google-auth-oauthlib>=0.4.0",
]

setup(
    name="rca_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "google": google_api_deps,
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0", "black>=20.8b1"],
        "all": google_api_deps,
    },
    entry_points={
        "console_scripts": [
            "rca-tools=rca_package.cli:main",
        ],
    },
    description="Root Cause Analysis tools for metric anomalies",
    author="Analytics Team",
    author_email="analytics@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
    include_package_data=True,
) 