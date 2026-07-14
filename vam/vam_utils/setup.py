"""
RAPP Lab 04 - Vision-Action Model Utilities Package
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="vam_utils",
    version="0.1.0",
    author="Dr. Maleen Jayasuriya",
    author_email="maleen@example.com",
    description="Vision-Action Model utilities for human-robot ensemble performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RAPP_LAB_04",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies (light, since most are in requirements.txt)
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "mypy>=1.0",
        ],
        "full": [
            # All dependencies for complete functionality
            "torch>=2.1.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "plotly>=5.15.0",
            "yourdfpy>=0.0.56",
            "roboticstoolbox-python>=1.1.0",
            "rosbags>=0.9.16",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI tools can be added here
            # "vam-process=vam_utils.cli:process_rosbags",
        ],
    },
    include_package_data=True,
    package_data={
        "vam_utils": ["*.yaml", "*.json"],
    },
)
