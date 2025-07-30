# Castl: A Consensus Framework for Robust Identification of Spatially Variable Genes in Spatial Transcriptomics

## 1 Overview
`Castl` is a novel consensus-based analytical framework designed to enhance the accuracy and robustness of spatially variable genes identification for spatially resolved transcriptomics through statistically rigorous algorithms, including **rank aggregation**, **p-value aggregation**, and **Stabl aggregation**. Comprehensive evaluations on both simulated and real-world data demonstrate that Castl consistently identifies biologically meaningful spatial expression patterns, mitigates method-specific biases and effectively controls FDRs across various biological contexts, resolutions, and spatial technologies. This flexible, assumption-free framework offers a robust and standardized foundation for spatially informed feature discovery in complex biological systems. 

![figure1](./docs/figures/Figure1_workflow.png)

## 2 System Requirements
### Python
- Python >= 3.9.5
- pandas >= 1.3.0
- numpy >= 1.21.0
- rpy2 >= 3.5.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- anndata >= 0.8.0
- scanpy >= 1.9.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scikit-learn >= 1.0.0

### R
- R >= 4.0.5
- dplyr >= 1.0.0
- tidyverse >= 1.3.0
- clusterProfiler >= 3.18.0
- org.Hs.eg.db >= 3.12.0
- patchwork >= 1.1.0
- ggplot2 >= 3.3.0
- TissueEnrich >= 1.8.0
- SummarizedExperiment >= 1.20.0

## 3 Installation

### Python
`Castl` can be installed directly from PyPI：
```bash
pip install Castl
```

or download from Github and install it:
```bash
git clone https://github.com/TheY11/Castl

cd Castl
pip install -e .
```

### R
We also provide the R package `castlRUtils` for calculating quality scores (QS) of SVGs.
```{r}
library(devtools)
devtools::install_github("TheY11/Castl", subdir = "Castl/r_utils", force = TRUE)
library(castlRUtils)
```

## 4 Tutorials
Detailed usage instructions and tutorials for `Castl` are available at:

[Tutorial 1: 10x Visium colorectal cancer liver metastasis datasets.](https://github.com/TheY11/Castl/docs/tutorials/Tutorial1_CRC.ipynb)

[Tutorial 2: 10x Visium human dorsolateral prefrontal cortex (DLPFC) datasets.](https://github.com/TheY11/Castl/docs/tutorials/Tutorial2_DLPFC.ipynb)

[Tutorial 3: Stereo-seq mouse olfactory bulb datasets.](https://github.com/TheY11/Castl/docs/tutorials/Tutorial3_Stereoseq_MOB.ipynb)

[Tutorial 4: Slide-seqV2 mouse olfactory bulb datasets.](https://github.com/TheY11/Castl/docs/tutorials/Tutorial4_SlideseqV2_MOB.ipynb)

[Tutorial 5: MERFISH mouse hypothalamic preoptic region data.](https://github.com/TheY11/Castl/docs/tutorials/Tutorial5_MERFISH.ipynb)

## 5 Improvements
For questions or issues, please [open an issue](https://github.com/TheY11/Castl/issues).