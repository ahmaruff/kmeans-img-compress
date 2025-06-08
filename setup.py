"""
Setup script for kmeans-img-compress package
"""

import os
from setuptools import setup, find_packages

# Read long description from README if exists
def read_long_description():
    """Read README file for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "A simple image compression library using K-Means clustering to reduce colors."

# Package metadata
setup(
    # Basic package info
    name="kmeans-img-compress",
    version="0.1.0",
    author="Ahmad Ma'ruf",
    author_email="ahmadmaruf2701@gmail.com",
    description="Simple image compression using K-Means clustering",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",

    # URLs
    url="https://github.com/ahmaruff/kmeans-img-compress",
    project_urls={
        "Bug Reports": "https://github.com/ahmaruff/kmeans-img-compress/issues",
        "Source": "https://github.com/ahmaruff/kmeans-img-compress",
    },

    # Package discovery
    packages=find_packages(),

    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        "scikit-learn>=1.0.0",
    ],

    # Optional dependencies for development
    # extras_require={
    #     "dev": [
    #         "pytest>=6.0",
    #         "pytest-cov",
    #         "black",
    #         "flake8",
    #     ],
    # },

    # Python version requirement
    python_requires=">=3.7",

    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Keywords for PyPI search
    keywords="image compression kmeans clustering color reduction",

    # Include additional files
    include_package_data=True,

    # Console scripts (optional - untuk CLI)
    entry_points={
        "console_scripts": [
            "kmeans-compress=kmeans_img_compress.core:main",  # Optional CLI
        ],
    },

    # License
    license="MIT",

    # Zip safe
    zip_safe=False,
)
