"""
Castl - a Consensus Framework for Robust Identification of Spatially Variable Genes in Spatial Transcriptomics

A consensus-based framework for SVG identification that integrates multiple SVG detection methods through three specialized modules: 
rank aggregation, p-value aggregation, and Stabl aggregation.
"""

from .preprocess import *
from .utils import *
from .castl import *

__version__ = "0.1.0"
