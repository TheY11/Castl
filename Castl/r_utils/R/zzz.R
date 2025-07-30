.onAttach <- function(libname, pkgname) {
  # 定义所有需要的包
  required_pkgs <- c("dplyr", "ggplot2", "clusterProfiler", "org.Hs.eg.db", 
                     "patchwork", "tidyr", "purrr", "tibble", "rlang", "stringr",
                     "TissueEnrich", "SummarizedExperiment")
  
  # 检查缺失的包
  missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
  
  # 如果有缺失的包，提示用户安装
  if (length(missing_pkgs) > 0) {
    packageStartupMessage(
      "The following packages are required but not installed:\n",
      paste("-", missing_pkgs, collapse = "\n"),
      "\nPlease install them with: install.packages(c('", 
      paste(missing_pkgs, collapse = "', '"), "'))",
      "\nFor Bioconductor packages, use: BiocManager::install(c('",
      paste(intersect(missing_pkgs, c("clusterProfiler", "org.Hs.eg.db", "TissueEnrich", "SummarizedExperiment")), 
            collapse = "', '"), "'))"
    )
  }
  
  # 静默加载核心包
  suppressPackageStartupMessages({
    library(dplyr)
    library(ggplot2)
    library(clusterProfiler)
    library(org.Hs.eg.db)
    library(patchwork)
    library(TissueEnrich)
    library(SummarizedExperiment)
  })
  
  # 显示加载完成信息
  packageStartupMessage(
    "castlRUtils package loaded.\n",
    "Required packages are now available:\n",
    "- Core packages: dplyr, ggplot2\n",
    "- Analysis packages: clusterProfiler, org.Hs.eg.db, patchwork, TissueEnrich, SummarizedExperiment\n",
    "All dependencies are ready to use."
  )
}