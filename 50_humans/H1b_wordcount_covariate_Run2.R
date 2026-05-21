# Script: H1b_wordcount_covariate_Run2.R
# Tests H1b convergent validity with word count as additional covariate
# Controls: gender, education, age, wc_total
# Also tests word count alone as control for comparison
#
# Input:  Data_Complete_Run2_wc.csv
# Output: results tables saved to OUTPUT_DIR

library(dplyr)
library(ppcor)

# --- Configuration ---
INPUT_FILE <- "Data_Complete_Run2_wc.csv"
RUN_LABEL  <- "Run2_Son46_Tnull_PrD_wc_covariate"
OUTPUT_DIR <- "/Users/seb/Library/CloudStorage/Dropbox/Akademin/Master Psykology/Master thesis project course/Thesis/Code/R-MTSebastian/60_Plots_Sebastian"
N_BOOT     <- 5000
set.seed(42)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

save_table <- function(df, name) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".csv"))
  write.table(df, filename, sep = "|", row.names = FALSE, quote = FALSE)
  cat(sprintf("  Table saved: %s\n", basename(filename)))
}

# --- Load data ---
df <- read.table(INPUT_FILE, header = TRUE, sep = "|",
                 stringsAsFactors = FALSE)
cat("Participants loaded:", nrow(df), "\n\n")

# --- Compute composites ---
df$Q_aut  <- rowMeans(df[, c("aut1","aut2","aut3","aut4")],        na.rm = TRUE)
df$Q_comp <- rowMeans(df[, c("comp1","comp2","comp3","comp4")],    na.rm = TRUE)
df$Q_rel  <- rowMeans(df[, c("rel1","rel2","rel3","rel4","rel5")], na.rm = TRUE)

df$LLM_aut  <- rowMeans(df[, c("llm_aut_rep1","llm_aut_rep2","llm_aut_rep3")], na.rm = TRUE)
df$LLM_comp <- rowMeans(df[, c("llm_com_rep1","llm_com_rep2","llm_com_rep3")], na.rm = TRUE)
df$LLM_rel  <- rowMeans(df[, c("llm_rel_rep1","llm_rel_rep2","llm_rel_rep3")], na.rm = TRUE)

# --- Compute age ---
df$age <- 2026 - df$birthyear

# --- Fisher z CI ---
fisher_ci <- function(r, n, k, conf = 0.95) {
  z    <- 0.5 * log((1 + r) / (1 - r))
  se   <- 1 / sqrt(n - k - 3)
  crit <- qnorm(1 - (1 - conf) / 2)
  lo   <- tanh(z - crit * se)
  hi   <- tanh(z + crit * se)
  c(lower = lo, upper = hi)
}

# --- Bootstrap partial correlation CI ---
boot_pcor_ci <- function(data, x_var, y_var, z_vars,
                         n_boot = N_BOOT, conf = 0.95) {
  boot_r <- replicate(n_boot, {
    idx    <- sample(nrow(data), replace = TRUE)
    d_boot <- data[idx, ]
    result <- pcor.test(d_boot[[x_var]], d_boot[[y_var]],
                        d_boot[, z_vars])
    result$estimate
  })
  quantile(boot_r, c((1 - conf) / 2, 1 - (1 - conf) / 2),
           na.rm = TRUE)
}

# --- Run partial correlation ---
run_partial_h1b <- function(q_var, llm_var, label, data, z_vars) {
  vars <- c(q_var, llm_var, z_vars)
  d    <- data[complete.cases(data[, vars]), vars]
  n    <- nrow(d)
  k    <- length(z_vars)
  
  r_zero  <- cor(d[[q_var]], d[[llm_var]], method = "pearson")
  ci_zero <- fisher_ci(r_zero, n, k = 0)
  
  pc             <- pcor.test(d[[q_var]], d[[llm_var]], d[, z_vars])
  r_part         <- pc$estimate
  p_part         <- pc$p.value
  ci_part_fisher <- fisher_ci(r_part, n, k)
  ci_part_boot   <- boot_pcor_ci(d, q_var, llm_var, z_vars)
  
  cat(sprintf("%s | N = %d | Controls: %s\n",
              label, n, paste(z_vars, collapse = ", ")))
  cat(sprintf("  Zero-order r = %.3f | Fisher CI [%.3f, %.3f]\n",
              r_zero, ci_zero[1], ci_zero[2]))
  cat(sprintf("  Partial r    = %.3f | p = %.3f\n", r_part, p_part))
  cat(sprintf("  Fisher CI    [%.3f, %.3f]\n",
              ci_part_fisher[1], ci_part_fisher[2]))
  cat(sprintf("  Bootstrap CI [%.3f, %.3f]\n\n",
              ci_part_boot[1], ci_part_boot[2]))
  
  data.frame(
    dimension    = label,
    n            = n,
    n_controls   = k,
    controls     = paste(z_vars, collapse = ", "),
    r_zero       = round(r_zero,          3),
    ci_zero_lo   = round(ci_zero[1],      3),
    ci_zero_hi   = round(ci_zero[2],      3),
    r_partial    = round(r_part,          3),
    p_partial    = round(p_part,          3),
    ci_fisher_lo = round(ci_part_fisher[1], 3),
    ci_fisher_hi = round(ci_part_fisher[2], 3),
    ci_boot_lo   = round(ci_part_boot[1],   3),
    ci_boot_hi   = round(ci_part_boot[2],   3)
  )
}

# ============================================================
# H1b: WORD COUNT AS COVARIATE
# ============================================================
cat("=== H1b: Convergent Validity — Word Count Covariate ===\n\n")

dimensions <- list(
  list(q = "Q_aut",  llm = "LLM_aut",  label = "Autonomy"),
  list(q = "Q_comp", llm = "LLM_comp", label = "Competence"),
  list(q = "Q_rel",  llm = "LLM_rel",  label = "Relatedness")
)

control_sets <- list(
  "wc_only"      = "wc_total",
  "demo_only"    = c("gender", "education", "age"),
  "demo_and_wc"  = c("gender", "education", "age", "wc_total")
)

results_list <- list()

for (dim in dimensions) {
  for (ctrl_name in names(control_sets)) {
    ctrl_vars <- control_sets[[ctrl_name]]
    result    <- run_partial_h1b(dim$q, dim$llm, dim$label,
                                 df, ctrl_vars)
    result$control_set <- ctrl_name
    results_list[[length(results_list) + 1]] <- result
  }
}

h1b_wc <- bind_rows(results_list)

print(h1b_wc)
save_table(h1b_wc, "H1b_wordcount_covariate")

cat(sprintf("\nDone. All outputs saved to %s\n", OUTPUT_DIR))