# Script: H1a_H1c+Exploratory_Hypothesis_Tests.R
# Tests H1a (categorical validity), H1b (convergent validity),
# H1c (discriminant validity), Exploratory aim (structural consistency)
#
# Input:  Data_Complete_Run1_cw.csv
# Output: results tables and plots saved to OUTPUT_DIR
#
# Deviations from preregistration:
# - H1c: Steiger's method used instead of Fisher r-to-z (correlations are dependent)
# - Exploratory aim: No inferential test applied (N=3 data points makes SE undefined)
# - All tests two-tailed (preregistered as one-tailed)

library(dplyr)
library(ggplot2)
library(cocor)
library(patchwork)
library(scales)
library(reshape2)

# --- Configuration ---
INPUT_FILE  <- "Data_Complete_Run2_wc.csv"
RUN_LABEL   <- ""
OUTPUT_DIR  <- "/Users/seb/Library/CloudStorage/Dropbox/Akademin/Master Psykology/Master thesis project course/Thesis/Code/R-MTSebastian/60_Plots_Sebastian"
N_BOOT      <- 10000
set.seed(42)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

save_table <- function(df, name) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".csv"))
  write.table(df, filename, sep = "|", row.names = FALSE, quote = FALSE)
  cat(sprintf("  Table saved: %s\n", basename(filename)))
}

save_plot <- function(plot, name, width = 8, height = 5) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".png"))
  ggsave(filename, plot = plot, width = width, height = height, dpi = 300)
  cat(sprintf("  Plot saved: %s\n", basename(filename)))
}

# --- Load data ---
df <- read.table(INPUT_FILE, header = TRUE, sep = "|",
                 stringsAsFactors = FALSE)
cat("Participants loaded:", nrow(df), "\n\n")

# --- Compute composites ---
# Questionnaire composites 
df$Q_aut  <- rowMeans(df[, c("aut1","aut2","aut3","aut4")],        na.rm = TRUE)
df$Q_comp <- rowMeans(df[, c("comp1","comp2","comp3","comp4")],    na.rm = TRUE)
df$Q_rel  <- rowMeans(df[, c("rel1","rel2","rel3","rel4","rel5")], na.rm = TRUE)

# LLM composites from repetition-level composites
df$LLM_aut  <- rowMeans(df[, c("llm_aut_rep1","llm_aut_rep2","llm_aut_rep3")], na.rm = TRUE)
df$LLM_comp <- rowMeans(df[, c("llm_com_rep1","llm_com_rep2","llm_com_rep3")], na.rm = TRUE)
df$LLM_rel  <- rowMeans(df[, c("llm_rel_rep1","llm_rel_rep2","llm_rel_rep3")], na.rm = TRUE)

N <- nrow(df)

# ============================================================
# H1a: CATEGORICAL VALIDITY
# Median split on Q scores, compare LLM scores between groups
# ============================================================
cat("=== H1a: Categorical Validity ===\n\n")

run_h1a <- function(q_var, llm_var, label) {
  q   <- df[[q_var]]
  llm <- df[[llm_var]]
  
  med       <- median(q, na.rm = TRUE)
  group     <- ifelse(q > med, "High", ifelse(q < med, "Low", NA))
  high_llm  <- llm[group == "High" & !is.na(group)]
  low_llm   <- llm[group == "Low"  & !is.na(group)]
  n_ties    <- sum(q == med, na.rm = TRUE)  # count exact ties excluded
  
  # Welch t-test
  welch   <- t.test(high_llm, low_llm, var.equal = FALSE)
  # Student t-test
  student <- t.test(high_llm, low_llm, var.equal = TRUE)
  
  cat(sprintf("%s | Median = %.3f | N high = %d, N low = %d, N ties excluded = %d\n",
              label, med, length(high_llm), length(low_llm), n_ties))
  cat(sprintf("  Mean LLM high = %.3f, low = %.3f, diff = %.3f\n",
              mean(high_llm), mean(low_llm),
              mean(high_llm) - mean(low_llm)))
  cat(sprintf("  Welch:   t(%.1f) = %.3f, p = %.3f, 95%% CI [%.3f, %.3f]\n",
              welch$parameter, welch$statistic, welch$p.value,
              welch$conf.int[1], welch$conf.int[2]))
  cat(sprintf("  Student: t(%.1f) = %.3f, p = %.3f, 95%% CI [%.3f, %.3f]\n\n",
              student$parameter, student$statistic, student$p.value,
              student$conf.int[1], student$conf.int[2]))
  
  data.frame(
    dimension   = label,
    median_Q    = round(med, 3),
    n_high      = length(high_llm),
    n_low       = length(low_llm),
    n_ties      = n_ties,
    mean_high   = round(mean(high_llm), 3),
    mean_low    = round(mean(low_llm),  3),
    diff        = round(mean(high_llm) - mean(low_llm), 3),
    welch_t     = round(welch$statistic,   3),
    welch_df    = round(welch$parameter,   1),
    welch_p     = round(welch$p.value,     3),
    welch_ci_lo = round(welch$conf.int[1], 3),
    welch_ci_hi = round(welch$conf.int[2], 3),
    student_t   = round(student$statistic,   3),
    student_df  = round(student$parameter,   1),
    student_p   = round(student$p.value,     3),
    student_ci_lo = round(student$conf.int[1], 3),
    student_ci_hi = round(student$conf.int[2], 3)
  )
}

h1a_results <- bind_rows(
  run_h1a("Q_aut",  "LLM_aut",  "Autonomy"),
  run_h1a("Q_comp", "LLM_comp", "Competence"),
  run_h1a("Q_rel",  "LLM_rel",  "Relatedness")
)

save_table(h1a_results, "H1a_categorical_validity")

# --- H1a plot: LLM score by group per dimension ---
h1a_long <- bind_rows(
  data.frame(dimension = "Autonomy",
             group = ifelse(df$Q_aut  > median(df$Q_aut,  na.rm=TRUE), "High", "Low"),
             llm   = df$LLM_aut),
  data.frame(dimension = "Competence",
             group = ifelse(df$Q_comp > median(df$Q_comp, na.rm=TRUE), "High", "Low"),
             llm   = df$LLM_comp),
  data.frame(dimension = "Relatedness",
             group = ifelse(df$Q_rel  > median(df$Q_rel,  na.rm=TRUE), "High", "Low"),
             llm   = df$LLM_rel)
) %>% filter(!is.na(group))

p_h1a <- ggplot(h1a_long, aes(x = group, y = llm, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ dimension) +
  scale_y_continuous(breaks = 1:5) +
  labs(title = paste("H1a: LLM scores by BPNS group —", RUN_LABEL),
       x = "BPNS group (median split)", y = "LLM score") +
  theme_minimal() +
  theme(legend.position = "none")
save_plot(p_h1a, "H1a_categorical_validity_boxplot", width = 10, height = 4)

# ============================================================
# H1b: CONVERGENT VALIDITY
# Pearson + Spearman correlations, Fisher z CI + Bootstrap CI
# ============================================================
cat("=== H1b: Convergent Validity ===\n\n")

# Fisher z CI function
fisher_ci <- function(r, n, conf = 0.95) {
  z    <- 0.5 * log((1 + r) / (1 - r))
  se   <- 1 / sqrt(n - 3)
  crit <- qnorm(1 - (1 - conf) / 2)
  lo   <- tanh(z - crit * se)
  hi   <- tanh(z + crit * se)
  c(lower = lo, upper = hi)
}

# Bootstrap CI function
boot_cor_ci <- function(x, y, method = "pearson", n_boot = N_BOOT, conf = 0.95) {
  complete <- complete.cases(x, y)
  x <- x[complete]; y <- y[complete]
  boot_r <- replicate(n_boot, {
    idx <- sample(length(x), replace = TRUE)
    cor(x[idx], y[idx], method = method)
  })
  quantile(boot_r, c((1 - conf) / 2, 1 - (1 - conf) / 2))
}

run_h1b <- function(q_var, llm_var, label) {
  x <- df[[q_var]];   y <- df[[llm_var]]
  n <- sum(complete.cases(x, y))
  
  r_p  <- cor(x, y, use = "complete.obs", method = "pearson")
  r_s  <- cor(x, y, use = "complete.obs", method = "spearman")
  
  ci_p_fisher <- fisher_ci(r_p, n)
  ci_s_fisher <- fisher_ci(r_s, n)
  ci_p_boot   <- boot_cor_ci(x, y, method = "pearson")
  ci_s_boot   <- boot_cor_ci(x, y, method = "spearman")
  
  cat(sprintf("%s | N = %d\n", label, n))
  cat(sprintf("  Pearson  r = %.3f | Fisher CI [%.3f, %.3f] | Boot CI [%.3f, %.3f]\n",
              r_p, ci_p_fisher[1], ci_p_fisher[2], ci_p_boot[1], ci_p_boot[2]))
  cat(sprintf("  Spearman r = %.3f | Fisher CI [%.3f, %.3f] | Boot CI [%.3f, %.3f]\n\n",
              r_s, ci_s_fisher[1], ci_s_fisher[2], ci_s_boot[1], ci_s_boot[2]))
  
  data.frame(
    dimension       = label,
    n               = n,
    pearson_r       = round(r_p, 3),
    pearson_fisher_lo = round(ci_p_fisher[1], 3),
    pearson_fisher_hi = round(ci_p_fisher[2], 3),
    pearson_boot_lo   = round(ci_p_boot[1],   3),
    pearson_boot_hi   = round(ci_p_boot[2],   3),
    spearman_r        = round(r_s, 3),
    spearman_fisher_lo = round(ci_s_fisher[1], 3),
    spearman_fisher_hi = round(ci_s_fisher[2], 3),
    spearman_boot_lo   = round(ci_s_boot[1],   3),
    spearman_boot_hi   = round(ci_s_boot[2],   3)
  )
}

h1b_results <- bind_rows(
  run_h1b("Q_aut",  "LLM_aut",  "Autonomy"),
  run_h1b("Q_comp", "LLM_comp", "Competence"),
  run_h1b("Q_rel",  "LLM_rel",  "Relatedness")
)

save_table(h1b_results, "H1b_convergent_validity")

# --- H1b plot: scatter per dimension ---
h1b_long <- bind_rows(
  data.frame(dimension="Autonomy",   Q=df$Q_aut,  LLM=df$LLM_aut),
  data.frame(dimension="Competence", Q=df$Q_comp, LLM=df$LLM_comp),
  data.frame(dimension="Relatedness",Q=df$Q_rel,  LLM=df$LLM_rel)
)

p_h1b <- ggplot(h1b_long, aes(x = Q, y = LLM)) +
  geom_point(alpha = 0.4, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, color = "steelblue") +
  facet_wrap(~ dimension) +
  labs(title = paste("H1b: Convergent validity — Q vs LLM scores —", RUN_LABEL),
       x = "Questionnaire composite", y = "LLM mean score") +
  theme_minimal()
save_plot(p_h1b, "H1b_convergent_validity_scatter", width = 10, height = 4)

# ============================================================
# H1c: DISCRIMINANT VALIDITY
# Steiger's test for dependent correlations
# ============================================================
cat("=== H1c: Discriminant Validity ===\n\n")

run_steiger <- function(llm_var, q_target, q_other, label_target, label_other) {
  x  <- df[[llm_var]]
  y1 <- df[[q_target]]
  y2 <- df[[q_other]]
  
  complete <- complete.cases(x, y1, y2)
  x  <- x[complete]; y1 <- y1[complete]; y2 <- y2[complete]
  n  <- length(x)
  
  # --- Pearson ---
  r_within <- cor(x, y1)
  r_cross  <- cor(x, y2)
  r_y1y2   <- cor(y1, y2)
  
  # --- Spearman + CIs ---
  r_within_s       <- cor(x, y1, method = "spearman")
  r_cross_s        <- cor(x, y2, method = "spearman")
  ci_within_s_boot <- boot_cor_ci(x, y1, method = "spearman")
  ci_cross_s_boot  <- boot_cor_ci(x, y2, method = "spearman")
  ci_within_s_fish <- fisher_ci(r_within_s, n)
  ci_cross_s_fish  <- fisher_ci(r_cross_s,  n)
  
  # --- Steiger test ---
  result <- cocor.dep.groups.overlap(
    r.jk = r_within, r.jh = r_cross, r.kh = r_y1y2,
    n = n, alternative = "two.sided", test = "steiger1980"
  )
  
  diff <- r_within - r_cross
  z1   <- 0.5 * log((1 + r_within) / (1 - r_within))
  z2   <- 0.5 * log((1 + r_cross)  / (1 - r_cross))
  se   <- sqrt(2 / (n - 3))
  crit <- qnorm(0.975)
  ci_lo <- tanh((z1 - z2) - crit * se)
  ci_hi <- tanh((z1 - z2) + crit * se)
  p_val <- result@steiger1980$p.value
  
  cat(sprintf("  %s vs %s | r_within = %.3f, r_cross = %.3f, diff = %.3f\n",
              label_target, label_other, r_within, r_cross, diff))
  cat(sprintf("  Steiger z = %.3f, p = %.3f | 95%% CI diff [%.3f, %.3f]\n",
              result@steiger1980$statistic, p_val, ci_lo, ci_hi))
  cat(sprintf("  Spearman: r_within = %.3f [%.3f, %.3f] (boot) | r_cross = %.3f [%.3f, %.3f] (boot)\n\n",
              r_within_s, ci_within_s_boot[1], ci_within_s_boot[2],
              r_cross_s,  ci_cross_s_boot[1],  ci_cross_s_boot[2]))
  
  data.frame(
    llm_dimension      = llm_var,
    comparison         = paste(label_target, "vs", label_other),
    r_within           = round(r_within,            3),
    r_cross            = round(r_cross,             3),
    diff               = round(diff,                3),
    ci_lo              = round(ci_lo,               3),
    ci_hi              = round(ci_hi,               3),
    steiger_z          = round(result@steiger1980$statistic, 3),
    p_value            = round(p_val,               3),
    n                  = n,
    sp_within          = round(r_within_s,          3),
    sp_within_fish_lo  = round(ci_within_s_fish[1], 3),
    sp_within_fish_hi  = round(ci_within_s_fish[2], 3),
    sp_within_boot_lo  = round(ci_within_s_boot[1], 3),
    sp_within_boot_hi  = round(ci_within_s_boot[2], 3),
    sp_cross           = round(r_cross_s,           3),
    sp_cross_fish_lo   = round(ci_cross_s_fish[1],  3),
    sp_cross_fish_hi   = round(ci_cross_s_fish[2],  3),
    sp_cross_boot_lo   = round(ci_cross_s_boot[1],  3),
    sp_cross_boot_hi   = round(ci_cross_s_boot[2],  3)
  )
}

h1c_results <- bind_rows(
  run_steiger("LLM_aut",  "Q_aut",  "Q_comp", "Q_Aut",  "Q_Comp"),
  run_steiger("LLM_aut",  "Q_aut",  "Q_rel",  "Q_Aut",  "Q_Rel"),
  run_steiger("LLM_comp", "Q_comp", "Q_aut",  "Q_Comp", "Q_Aut"),
  run_steiger("LLM_comp", "Q_comp", "Q_rel",  "Q_Comp", "Q_Rel"),
  run_steiger("LLM_rel",  "Q_rel",  "Q_aut",  "Q_Rel",  "Q_Aut"),
  run_steiger("LLM_rel",  "Q_rel",  "Q_comp", "Q_Rel",  "Q_Comp")
)

print(h1c_results)
save_table(h1c_results, "H1c_discriminant_validity")

# ============================================================
# Exploratory Aim: STRUCTURAL CONSISTENCY
# Meta-correlation of inter-need correlation patterns
# ============================================================
cat("=== H1d: Structural Consistency ===\n\n")

# Inter-need correlations for questionnaire
r_q_ac <- cor(df$Q_aut,  df$Q_comp, use = "complete.obs")
r_q_ar <- cor(df$Q_aut,  df$Q_rel,  use = "complete.obs")
r_q_cr <- cor(df$Q_comp, df$Q_rel,  use = "complete.obs")

# Inter-need correlations for LLM
r_l_ac <- cor(df$LLM_aut,  df$LLM_comp, use = "complete.obs")
r_l_ar <- cor(df$LLM_aut,  df$LLM_rel,  use = "complete.obs")
r_l_cr <- cor(df$LLM_comp, df$LLM_rel,  use = "complete.obs")

q_vec   <- c(r_q_ac, r_q_ar, r_q_cr)
llm_vec <- c(r_l_ac, r_l_ar, r_l_cr)

r_meta <- cor(q_vec, llm_vec)

cat("Inter-need correlations:\n")
cat(sprintf("  Q:   Aut-Comp = %.3f, Aut-Rel = %.3f, Comp-Rel = %.3f\n",
            r_q_ac, r_q_ar, r_q_cr))
cat(sprintf("  LLM: Aut-Comp = %.3f, Aut-Rel = %.3f, Comp-Rel = %.3f\n",
            r_l_ac, r_l_ar, r_l_cr))
cat(sprintf("  Meta-correlation r = %.3f (N=3, descriptive only)\n\n", r_meta))

h1d_interneed <- data.frame(
  pair       = c("Aut-Comp", "Aut-Rel", "Comp-Rel"),
  r_Q        = round(q_vec,   3),
  r_LLM      = round(llm_vec, 3)
)

h1d_meta <- data.frame(
  meta_r            = round(r_meta, 3),
  note              = "N=3 data points — descriptive only, no inferential test applied"
)

print(h1d_interneed)
cat(sprintf("\nMeta-correlation: r = %.3f\n\n", r_meta))
save_table(h1d_interneed, "H1d_structural_consistency_interneed")
save_table(h1d_meta,      "H1d_structural_consistency_metacorr")

# --- H1d plot: inter-need correlation comparison ---
h1d_long <- data.frame(
  pair   = rep(c("Aut-Comp","Aut-Rel","Comp-Rel"), 2),
  method = rep(c("Questionnaire","LLM"), each = 3),
  r      = c(q_vec, llm_vec)
)

p_h1d <- ggplot(h1d_long, aes(x = pair, y = r, fill = method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_y_continuous(breaks = seq(-1, 1, 0.25)) +
  labs(title = paste("H1d: Inter-need correlations by method —", RUN_LABEL),
       x = "Need pair", y = "Pearson r", fill = "Method") +
  theme_minimal()
save_plot(p_h1d, "H1d_structural_consistency_barplot")

cat(sprintf("\nDone. All outputs saved to %s\n", OUTPUT_DIR))


# ============================================================
# HEATMAPS: Validity visualisation
# ============================================================

# --- Colour palette ---
COL_DIFF      <- "#DDA613"
COL_AI_SHADES <- c("#1F5A71","#2085AB","#489CC1","#6DBCE0","#9ecae1","#FFFFFF")
COL_HU_SHADES <- c("#BE2A16","#CE5545","#DC7D71","#E9A39B","#F5C9C5","#FFFFFF")

# --- Fill scale: blue = negative, red = positive ---
fill_scale <- scale_fill_gradientn(
  colours = c(
    COL_AI_SHADES[1], COL_AI_SHADES[3], COL_AI_SHADES[5],
    "#FFFFFF",
    COL_HU_SHADES[5], COL_HU_SHADES[3], COL_HU_SHADES[1]
  ),
  values = rescale(c(-1, -0.5, -0.1, 0, 0.1, 0.5, 1)),
  limits = c(-1, 1),
  name   = "Pearson r"
)

# --- APA 7 theme ---
heatmap_theme <- theme_classic(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", size = 12, hjust = 0),
    plot.subtitle   = element_text(size = 9, color = "gray40", hjust = 0),
    axis.text.x     = element_text(size = 11, color = "black"),
    axis.text.y     = element_text(size = 11, color = "black"),
    axis.title.x    = element_text(size = 11, color = "black",
                                   margin = margin(t = 8)),
    axis.title.y    = element_text(size = 11, color = "black",
                                   margin = margin(r = 8)),
    axis.line       = element_blank(),
    axis.ticks      = element_blank(),
    legend.position = "right",
    legend.title    = element_text(size = 10),
    legend.text     = element_text(size = 9),
    panel.grid      = element_blank(),
    panel.border    = element_blank(),
    plot.margin     = margin(12, 12, 12, 12)
  )

# --- Descriptive label function ---
make_label <- function(r) {
  case_when(
    abs(r) >= 0.50 ~ paste0(sprintf("%.2f", r), "\nStrong"),
    abs(r) >= 0.30 ~ paste0(sprintf("%.2f", r), "\nModerate"),
    abs(r) >= 0.10 ~ paste0(sprintf("%.2f", r), "\nSmall"),
    TRUE           ~ paste0(sprintf("%.2f", r), "\nNegligible")
  )
}

# --- Diagonal borders for validity coefficients ---
diag_borders <- data.frame(
  xmin = c(0.5, 1.5, 2.5),
  xmax = c(1.5, 2.5, 3.5),
  ymin = c(0.5, 1.5, 2.5),
  ymax = c(1.5, 2.5, 3.5)
)

llm_dims <- c("LLM Autonomy", "LLM Competence", "LLM Relatedness")
q_dims   <- c("Q Relatedness",   "Q Competence",   "Q Autonomy")

# --- 3x3 plot function ---
make_3x3_plot <- function(r_values, title_text, subtitle_text) {
  df_plot <- expand.grid(
    LLM = factor(llm_dims, levels = rev(llm_dims)),
    Q   = factor(q_dims,   levels = q_dims)
  ) %>%
    mutate(
      r          = r_values,
      label      = make_label(r),
      text_color = ifelse(abs(r) > 0.4, "white", "black")
    )
  
  ggplot(df_plot, aes(x = Q, y = LLM, fill = r)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = label, color = text_color),
              size = 3.8, fontface = "bold", lineheight = 0.85) +
    scale_color_identity() +
    fill_scale +
    geom_rect(
      data        = diag_borders,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill        = NA, color = COL_DIFF, linewidth = 1.8,
      inherit.aes = FALSE
    ) +
    labs(title    = title_text,
         subtitle = subtitle_text,
         x        = "Questionnaire dimension",
         y        = "LLM dimension") +
    heatmap_theme
}

# Observed cross-method correlations
cor_matrix <- matrix(
  c(
    cor(df$LLM_aut,  df$Q_aut,  use = "complete.obs"),
    cor(df$LLM_aut,  df$Q_comp, use = "complete.obs"),
    cor(df$LLM_aut,  df$Q_rel,  use = "complete.obs"),
    cor(df$LLM_comp, df$Q_aut,  use = "complete.obs"),
    cor(df$LLM_comp, df$Q_comp, use = "complete.obs"),
    cor(df$LLM_comp, df$Q_rel,  use = "complete.obs"),
    cor(df$LLM_rel,  df$Q_aut,  use = "complete.obs"),
    cor(df$LLM_rel,  df$Q_comp, use = "complete.obs"),
    cor(df$LLM_rel,  df$Q_rel,  use = "complete.obs")
  ),
  nrow = 3, byrow = TRUE,
  dimnames = list(llm_dims, q_dims)
)

obs_r_a <- as.vector(t(cor_matrix[rev(llm_dims), ]))

ideal_r_a <- c(
  0.00, 0.00, 0.60,
  0.00, 0.60, 0.00,
  0.60, 0.00, 0.00
)

p_a_obs <- make_3x3_plot(
  r_values      = obs_r_a,
  title_text    = "Observed",
  subtitle_text = "Gold border = validity diagonal (H1b)\nOff-diagonal = cross-need correlations (H1c)"
)

p_a_ideal <- make_3x3_plot(
  r_values      = ideal_r_a,
  title_text    = "Ideal — H1b and H1c",
  subtitle_text = "Strong validity diagonal\nNegligible cross-need correlations"
)

p_a_combined <- (p_a_obs + p_a_ideal) +
  plot_annotation(
    title    = paste("H1b & H1c: Cross-method correlation matrix —", RUN_LABEL),
    subtitle = "Rows: LLM-derived scores  |  Columns: Questionnaire scores\nBlue = negative  |  Red = positive  |  White = negligible",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10, color = "gray40")
    )
  ) +
  plot_layout(guides = "collect")

save_plot(p_a_combined, "H1b_H1c_crossmethod_heatmap_combined",
          width = 18, height = 7)

# --- H1d heatmap function ---
pairs <- c("Autonomy–Competence", "Autonomy–Relatedness", "Competence–Relatedness")

make_h1d_plot <- function(r_q_vals, r_l_vals, title_text, subtitle_text) {
  df_plot <- data.frame(
    pair   = rep(factor(pairs, levels = pairs), 2),
    method = rep(c("Questionnaire", "LLM"), each = 3),
    r      = c(r_q_vals, r_l_vals)
  ) %>%
    mutate(
      label      = make_label(r),
      text_color = ifelse(abs(r) > 0.4, "white", "black")
    )
  
  df_plot$method <- factor(df_plot$method,
                           levels = c("LLM", "Questionnaire"))
  
  ggplot(df_plot, aes(x = pair, y = method, fill = r)) +
    geom_tile(color = "white", linewidth = 1.2) +
    geom_text(aes(label = label, color = text_color),
              size = 3.8, fontface = "bold", lineheight = 0.85) +
    scale_color_identity() +
    fill_scale +
    scale_y_discrete(limits = c("LLM", "Questionnaire")) +
    labs(title    = title_text,
         subtitle = subtitle_text,
         x        = "Need pair",
         y        = NULL) +
    heatmap_theme
}

p_b_obs <- make_h1d_plot(
  r_q_vals      = q_vec,
  r_l_vals      = llm_vec,
  title_text    = "Observed",
  subtitle_text = "H1d ideal if LLM row mirrors Questionnaire row"
)

p_b_ideal <- make_h1d_plot(
  r_q_vals      = q_vec,
  r_l_vals      = q_vec,
  title_text    = "Ideal — H1d",
  subtitle_text = "LLM inter-need structure mirrors questionnaire exactly"
)

p_b_combined <- (p_b_obs + p_b_ideal) +
  plot_annotation(
    title    = paste("H1d: Inter-need correlation patterns —", RUN_LABEL),
    subtitle = "Blue = negative  |  Red = positive  |  White = negligible",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10, color = "gray40")
    )
  ) +
  plot_layout(guides = "collect")

save_plot(p_b_combined, "H1d_interneed_heatmap_combined",
          width = 14, height = 5)

cat(sprintf("\nDone. All outputs saved to %s\n", OUTPUT_DIR))


# ============================================================
# H1d FOREST PLOT: Inter-need correlation patterns
# ============================================================

h1d_forest <- data.frame(
  pair   = rep(pairs, 2),
  method = rep(c("Questionnaire", "LLM"), each = 3),
  r      = c(q_vec, llm_vec),
  stringsAsFactors = FALSE
) %>%
  mutate(
    ci_lo  = mapply(function(rv, xv, yv) boot_cor_ci(xv, yv)[1],
                    r,
                    list(df$Q_aut,  df$Q_aut,  df$Q_comp,
                         df$LLM_aut, df$LLM_aut, df$LLM_comp),
                    list(df$Q_comp, df$Q_rel,  df$Q_rel,
                         df$LLM_comp, df$LLM_rel, df$LLM_rel)),
    ci_hi  = mapply(function(rv, xv, yv) boot_cor_ci(xv, yv)[2],
                    r,
                    list(df$Q_aut,  df$Q_aut,  df$Q_comp,
                         df$LLM_aut, df$LLM_aut, df$LLM_comp),
                    list(df$Q_comp, df$Q_rel,  df$Q_rel,
                         df$LLM_comp, df$LLM_rel, df$LLM_rel)),
    pair   = factor(pair, levels = rev(pairs)),
    method = factor(method, levels = c("Questionnaire", "LLM"))
  )
p_h1d_forest <- ggplot(h1d_forest,
                       aes(x = r, y = pair, color = method, shape = method)) +
  geom_vline(xintercept = 0, color = "gray50", linewidth = 0.4,
             linetype = "dashed") +
  geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi),
                 height = 0.15, linewidth = 0.7,
                 position = position_dodge(width = 0.45)) +
  geom_point(size = 3,
             position = position_dodge(width = 0.45)) +
  scale_x_continuous(
    limits = c(-0.35, 0.85),
    breaks = seq(-0.2, 0.8, 0.2),
    name   = expression(italic(r))
  ) +
  scale_color_manual(
    values = c("Questionnaire" = "#BE2A16", "LLM" = "#1F5A71"),
    name   = NULL
  ) +
  scale_shape_manual(
    values = c("Questionnaire" = 16, "LLM" = 17),
    name   = NULL
  ) +
  labs(y = NULL) +
  theme_minimal(base_family = "Calibri", base_size = 11) +
  theme(
    axis.text         = element_text(family = "Calibri", size = 15, color = "black"),
    axis.title.x      = element_text(family = "Calibri", size = 13),
    panel.grid.minor  = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position   = "bottom",
    legend.text       = element_text(family = "Calibri", size = 11),
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.margin       = margin(12, 12, 12, 12)
  )

save_plot(p_h1d_forest, "H1d_interneed_forest", width = 10, height = 4)


# ============================================================
# FIGURE 3: Full 6x6 Correlation Matrix (APA-compliant)
# ============================================================

vars_6  <- c("Q_aut", "Q_comp", "Q_rel", "LLM_aut", "LLM_comp", "LLM_rel")
cor_6x6 <- cor(df[, vars_6], use = "complete.obs")

var_names <- c(
  "Q.Autonomy", "Q.Competence", "Q.Relatedness",
  "LLM.Autonomy", "LLM.Competence", "LLM.Relatedness"
)
dimnames(cor_6x6) <- list(var_names, var_names)

df_cor      <- reshape2::melt(cor_6x6, varnames = c("Var1", "Var2"), value.name = "r")
df_cor$Var2 <- factor(df_cor$Var2, levels = var_names)
df_cor$Var1 <- factor(df_cor$Var1, levels = rev(var_names))

# Lower triangle only
orig_row_idx <- match(as.character(df_cor$Var1), var_names)
orig_col_idx <- match(as.character(df_cor$Var2), var_names)
df_cor       <- df_cor[orig_col_idx <= orig_row_idx, ]

# Plain dimension labels — bold group names added as annotations
x_labels <- c(
  "Autonomy\nQuestionnaire", "Competence\nQuestionnaire", "Relatedness\nQuestionnaire",
  "Autonomy\nLLM",           "Competence\nLLM", "Relatedness\nLLM"
)

y_labels <- c(
  "Relatedness\nLLM", "Competence\nLLM", "Autonomy\nLLM",
  "Relatedness\nQuestionnaire", "Competence\nQuestionnaire", "Autonomy\nQuestionnaire"
)

p_fig3 <- ggplot(df_cor, aes(x = Var2, y = Var1, fill = r)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_vline(xintercept = 3.5, color = "gray60", linewidth = 0.4) +
  geom_hline(yintercept = 3.5, color = "gray60", linewidth = 0.4) +
  geom_text(
    aes(label = sub("^0\\.", ".", sub("^-0\\.", "-.", sprintf("%.2f", r))),
        color = ifelse(r > 0.65, "white", "black")),
    family = "Calibri", size = 8 / .pt, na.rm = TRUE
  ) +
  scale_color_identity() +
  scale_x_discrete(labels = x_labels) +
  scale_y_discrete(labels = y_labels) +
  scale_fill_gradientn(
    colours = c(COL_HU_SHADES[6], COL_HU_SHADES[5], COL_HU_SHADES[4],
                COL_HU_SHADES[3], COL_HU_SHADES[2], COL_HU_SHADES[1]),
    values  = rescale(c(-0.10, 0.10, 0.30, 0.50, 0.75, 1)),
    limits  = c(-0.10, 1.001),
    na.value = "white",
    name    = expression(italic(r)),
    guide   = guide_colorbar(barwidth = 0.8, barheight = 8,
                             frame.colour = "black", ticks.colour = "black")
  ) +
 
  theme_minimal(base_family = "Calibri", base_size = 9) +
  theme(
    plot.title       = element_blank(),
    axis.title       = element_blank(),
    axis.text.x = element_text(
      family = "Calibri", size = 7, color = "black", hjust = 0.5,
      face = "plain", lineheight = 0.9
    ),
    axis.text.y = element_text(
      family = "Calibri", size = 7, color = "black",
      face = "plain", lineheight = 0.9
    ),
    panel.grid       = element_blank(),
    legend.title     = element_text(family = "Calibri", size = 9),
    legend.text      = element_text(family = "Calibri", size = 8),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(40, 12, 30, 12)   # more top/bottom for two-line labels
  )

ggsave(
  file.path(OUTPUT_DIR, "Figure3_CorrelationMatrix.png"),
  plot = p_fig3, width = 17, height = 14, units = "cm", dpi = 300
)
cat("  Figure 3 saved.\n")