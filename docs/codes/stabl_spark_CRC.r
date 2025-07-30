rm(list = ls())

library(SPARK)
library(tibble)
library(dplyr)

tissue_type <- "CRC"
dataset <- "colon2"

counts_matrix_path <- file.path("..", "data", "CRC", paste0("ST-", dataset), 
                                paste0(tissue_type, "_", dataset, "_combined_counts_matrix.csv"))
col_data_path <- file.path("..", "data", "CRC", paste0("ST-", dataset), 
                           paste0(tissue_type, "_", dataset, "_combined_coldata.csv"))

output_csv_path <- file.path("..", "results", "CRC", tissue_type, dataset, paste0(dataset,"_stabl"), 
                             paste0(tissue_type, "_", dataset, "_spark_stabl_processed.csv"))

if (!file.exists(counts_matrix_path)) {
  stop(paste("Counts matrix file not found:", counts_matrix_path))
}

if (!file.exists(col_data_path)) {
  stop(paste("Coldata file not found:", col_data_path))
}

counts <- read.csv(counts_matrix_path, header = TRUE, row.names = 1)
counts <- t(counts)
info <- read.csv(col_data_path, header = TRUE, row.names = 1)
info <- info[, 1:2]
colnames(info) <- c("x", "y")

print(paste("Data loaded for", dataset, "- Dimensions:", dim(counts)))

spark <- CreateSPARKObject(
  counts = counts, 
  location = info, 
  percentage = 0.1, 
  min_total_counts = 10
)

spark@lib_size <- apply(spark@counts, 2, sum)

print(paste("Running SPARK analysis for", dataset))
spark <- spark.vc(
  spark, 
  covariates = NULL, 
  lib_size = spark@lib_size, 
  num_core = 10, 
  verbose = TRUE, 
  fit.maxiter = 500
)
spark <- spark.test(spark, check_positive = TRUE, verbose = TRUE)

results <- spark@res_mtest
results <- rownames_to_column(results, var = "gene")

results <- results %>% rename(adjusted_p_value = adjusted_pvalue)
results <- results %>% mutate(pred = ifelse(adjusted_p_value < 0.05, 1, 0))
results <- results %>% mutate(gene = gsub("-", ".", gene))

write.csv(results, file = output_csv_path, row.names = TRUE)
print(paste("Completed SPARK analysis for", dataset))