from setuptools import setup, find_packages
import os
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def install_r_package():
    try:
        r_script = """
        if (!require("devtools")) install.packages("devtools")
        devtools::install("{r_pkg_path}")
        """.format(r_pkg_path=os.path.abspath("Castl/r_utils"))
        
        subprocess.run(["Rscript", "-e", r_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install R package: {e}")

install_r_package()

setup(
    name="STCastl",
    version="0.1.1",
    packages=["Castl"],
    package_dir={"": "."},
    author="Yiyi Yu",
    author_email="yiyiyu11@163.com",
    description="A Consensus Framework for Robust Identification of Spatially Variable Genes in Spatial Transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheY11/Castl",
    package_data={
        'Castl.r_utils': ['R/*.R', 'DESCRIPTION', 'NAMESPACE'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "rpy2>=3.5.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "anndata>=0.8.0",
        "scanpy>=1.9.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
            "sphinx>=4.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/TheY11/Castl/issues",
        "Source": "https://github.com/TheY11/Castl",
    },
    keywords=[
        "consensus",
        "spatially variable genes",
        "spatial transcriptomics",
    ],
)