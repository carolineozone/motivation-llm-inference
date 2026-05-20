# Script: Descriptives_and_Reliability.R
# Computes descriptive statistics and Reliability for the merged dataset
# Covers: sample demographics, BPNS questionnaire composites,
#         Cronbach's alpha per subscale, LLM score distributions, 
#         LLM score Reliability: ICC [2,3] + SD + Cl 95%
#         
#         ICC(2,3): two-way random, absolute agreement, average of 3 measures
#
# Input:  Data_Complete_Run1.csv
# Output: descriptives tables and plots saved to OUTPUT_DIR

library(dplyr)
library(psych)
library(ggplot2)
library(lme4)

# --- Configuration ---
INPUT_FILE  <- "Data_Complete_Run2_wc.csv"
RUN_LABEL   <- "Run2_Son46_Tnull_PrE"
OUTPUT_DIR  <- "/Users/seb/Library/CloudStorage/Dropbox/Akademin/Master Psykology/Master thesis project course/Thesis/Code/R-MTSebastian/60_Plots_Sebastian"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Helper: save plot with descriptive name
save_plot <- function(plot, name, width = 8, height = 5) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".png"))
  ggsave(filename, plot = plot, width = width, height = height, dpi = 300)
  cat(sprintf("  Plot saved: %s\n", basename(filename)))
}

# Helper: save dataframe with descriptive name
save_table <- function(df, name) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".csv"))
  write.table(df, filename, sep = "|", row.names = FALSE, quote = FALSE)
  cat(sprintf("  Table saved: %s\n", basename(filename)))
}

# --- Load data ---
df <- read.table(INPUT_FILE, header = TRUE, sep = "|",
                 stringsAsFactors = FALSE)
cat("Participants loaded:", nrow(df), "\n")

# ============================================================
# SECTION 1: DEMOGRAPHICS
# ============================================================
cat("\n=== Demographics ===\n\n")

df$age <- 2026 - df$birthyear

# Gender
gender_table <- df %>%
  count(gender) %>%
  mutate(pct = round(100 * n / sum(n), 1))
cat("Gender distribution:\n")
print(gender_table)
save_table(gender_table, "demographics_gender")

p_gender <- ggplot(gender_table, aes(x = factor(gender), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")),
            vjust = -0.5, size = 3.5) +
  labs(title = paste("Gender distribution —", RUN_LABEL),
       x = "Gender", y = "Count") +
  theme_minimal()
save_plot(p_gender, "demographics_gender_barplot")

# Age
cat(sprintf("\nAge: M = %.2f, SD = %.2f, range = %d-%d\n",
            mean(df$age, na.rm = TRUE), sd(df$age, na.rm = TRUE),
            min(df$age, na.rm = TRUE), max(df$age, na.rm = TRUE)))

age_summary <- data.frame(
  mean = round(mean(df$age, na.rm = TRUE), 2),
  sd   = round(sd(df$age,   na.rm = TRUE), 2),
  min  = min(df$age, na.rm = TRUE),
  max  = max(df$age, na.rm = TRUE)
)
save_table(age_summary, "demographics_age_summary")

p_age <- ggplot(df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", alpha = 0.8, color = "white") +
  labs(title = paste("Age distribution —", RUN_LABEL),
       x = "Age", y = "Count") +
  theme_minimal()
save_plot(p_age, "demographics_age_histogram")

# --- Age group distribution ---
df$age_group <- cut(df$age,
                    breaks = c(17, 29, 39, 49, 59, 100),
                    labels = c("18-29", "30-39", "40-49", "50-59", "60+"))

age_group_table <- df %>%
  count(age_group) %>%
  mutate(pct = round(100 * n / sum(n), 1))

cat("\nAge group distribution:\n")
print(age_group_table)
save_table(age_group_table, "demographics_age_groups")

# Education
edu_table <- df %>%
  count(education) %>%
  mutate(pct = round(100 * n / sum(n), 1))
cat("\nEducation distribution:\n")
print(edu_table)
save_table(edu_table, "demographics_education")

p_edu <- ggplot(edu_table, aes(x = factor(education), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
  geom_text(aes(label = paste0(n, " (", pct, "%)")),
            vjust = -0.5, size = 3.5) +
  labs(title = paste("Education distribution —", RUN_LABEL),
       x = "Education level", y = "Count") +
  theme_minimal()
save_plot(p_edu, "demographics_education_barplot")

# ============================================================
# SECTION 1B: WORD COUNT DESCRIPTIVES
# ============================================================
cat("\n=== Word Count Descriptives ===\n\n")

wc_summary <- data.frame(
  text      = c("Text 1", "Text 2", "Text 3", "Total"),
  n         = c(sum(!is.na(df$wc_text_1)), sum(!is.na(df$wc_text_2)),
                sum(!is.na(df$wc_text_3)), sum(!is.na(df$wc_total))),
  mean      = round(c(mean(df$wc_text_1, na.rm=TRUE), mean(df$wc_text_2, na.rm=TRUE),
                      mean(df$wc_text_3, na.rm=TRUE), mean(df$wc_total,  na.rm=TRUE)), 1),
  sd        = round(c(sd(df$wc_text_1,   na.rm=TRUE), sd(df$wc_text_2,   na.rm=TRUE),
                      sd(df$wc_text_3,   na.rm=TRUE), sd(df$wc_total,    na.rm=TRUE)), 1),
  min       = c(min(df$wc_text_1, na.rm=TRUE), min(df$wc_text_2, na.rm=TRUE),
                min(df$wc_text_3, na.rm=TRUE), min(df$wc_total,  na.rm=TRUE)),
  max       = c(max(df$wc_text_1, na.rm=TRUE), max(df$wc_text_2, na.rm=TRUE),
                max(df$wc_text_3, na.rm=TRUE), max(df$wc_total,  na.rm=TRUE))
)

print(wc_summary)
save_table(wc_summary, "wordcount_descriptives")

# --- Word count histogram per text ---
wc_long <- data.frame(
  text  = rep(c("Text 1","Text 2","Text 3"), each = nrow(df)),
  words = c(df$wc_text_1, df$wc_text_2, df$wc_text_3)
)

p_wc_hist <- ggplot(wc_long, aes(x = words)) +
  geom_histogram(binwidth = 10, fill = "steelblue", alpha = 0.8, color = "white") +
  facet_wrap(~ text) +
  labs(title = paste("Word count per text —", RUN_LABEL),
       x = "Word count", y = "Count") +
  theme_minimal()
save_plot(p_wc_hist, "wordcount_histogram", width = 10, height = 4)

# --- Total word count histogram ---
p_wc_total <- ggplot(df, aes(x = wc_total)) +
  geom_histogram(binwidth = 20, fill = "steelblue", alpha = 0.8, color = "white") +
  labs(title = paste("Total word count across 3 texts —", RUN_LABEL),
       x = "Total word count", y = "Count") +
  theme_minimal()
save_plot(p_wc_total, "wordcount_total_histogram")

# --- Boxplot per text ---
p_wc_box <- ggplot(wc_long, aes(x = text, y = words, fill = text)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = paste("Word count distribution per text —", RUN_LABEL),
       x = NULL, y = "Word count") +
  theme_minimal() +
  theme(legend.position = "none")
save_plot(p_wc_box, "wordcount_boxplot")

# ============================================================
# SECTION 2: BPNS QUESTIONNAIRE COMPOSITES + CRONBACH'S ALPHA
# ============================================================
cat("\n=== BPNS Questionnaire Scores ===\n\n")

aut_items  <- c("aut1",  "aut2",  "aut3",  "aut4")
comp_items <- c("comp1", "comp2", "comp3", "comp4")
rel_items  <- c("rel1",  "rel2",  "rel3",  "rel4",  "rel5")

df$Q_aut  <- rowMeans(df[, aut_items],  na.rm = TRUE)
df$Q_comp <- rowMeans(df[, comp_items], na.rm = TRUE)
df$Q_rel  <- rowMeans(df[, rel_items],  na.rm = TRUE)

alpha_aut  <- psych::alpha(df[, aut_items])$total$raw_alpha
alpha_comp <- psych::alpha(df[, comp_items])$total$raw_alpha
alpha_rel  <- psych::alpha(df[, rel_items])$total$raw_alpha

bpns_desc <- data.frame(
  subscale = c("Autonomy", "Competence", "Relatedness"),
  n     = c(sum(!is.na(df$Q_aut)),  sum(!is.na(df$Q_comp)),  sum(!is.na(df$Q_rel))),
  mean  = round(c(mean(df$Q_aut, na.rm=TRUE), mean(df$Q_comp, na.rm=TRUE), mean(df$Q_rel, na.rm=TRUE)), 3),
  sd    = round(c(sd(df$Q_aut,   na.rm=TRUE), sd(df$Q_comp,   na.rm=TRUE), sd(df$Q_rel,   na.rm=TRUE)), 3),
  min   = round(c(min(df$Q_aut,  na.rm=TRUE), min(df$Q_comp,  na.rm=TRUE), min(df$Q_rel,  na.rm=TRUE)), 3),
  max   = round(c(max(df$Q_aut,  na.rm=TRUE), max(df$Q_comp,  na.rm=TRUE), max(df$Q_rel,  na.rm=TRUE)), 3),
  alpha = round(c(alpha_aut, alpha_comp, alpha_rel), 3)
)

cat("BPNS descriptives:\n")
print(bpns_desc)
save_table(bpns_desc, "BPNS_descriptives_alpha")

cat(sprintf("\nCronbach's alpha:\n"))
cat(sprintf("  Autonomy:    %.3f\n", alpha_aut))
cat(sprintf("  Competence:  %.3f\n", alpha_comp))
cat(sprintf("  Relatedness: %.3f\n", alpha_rel))

# BPNS distribution plot
bpns_long <- data.frame(
  subscale = rep(c("Autonomy", "Competence", "Relatedness"), each = nrow(df)),
  score    = c(df$Q_aut, df$Q_comp, df$Q_rel)
)

p_bpns_hist <- ggplot(bpns_long, aes(x = score)) +
  geom_histogram(binwidth = 0.25, fill = "steelblue", alpha = 0.8, color = "white") +
  facet_wrap(~ subscale) +
  scale_x_continuous(breaks = 1:5) +
  labs(title = paste("BPNS composite score distributions —", RUN_LABEL),
       x = "Composite score", y = "Count") +
  theme_minimal()
save_plot(p_bpns_hist, "BPNS_composite_histogram", width = 10, height = 4)

p_bpns_box <- ggplot(bpns_long, aes(x = subscale, y = score, fill = subscale)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(breaks = 1:5) +
  labs(title = paste("BPNS composite score boxplots —", RUN_LABEL),
       x = NULL, y = "Composite score") +
  theme_minimal() +
  theme(legend.position = "none")
save_plot(p_bpns_box, "BPNS_composite_boxplot")

# ============================================================
# SECTION 3: LLM SCORE DISTRIBUTIONS
# ============================================================
cat("\n=== LLM Score Distributions ===\n\n")

df$LLM_aut  <- rowMeans(df[, c("llm_aut_rep1",  "llm_aut_rep2",  "llm_aut_rep3")],  na.rm = TRUE)
df$LLM_comp <- rowMeans(df[, c("llm_com_rep1", "llm_com_rep2", "llm_com_rep3")], na.rm = TRUE)
df$LLM_rel  <- rowMeans(df[, c("llm_rel_rep1",  "llm_rel_rep2",  "llm_rel_rep3")],  na.rm = TRUE)

llm_desc <- data.frame(
  dimension = c("Autonomy", "Competence", "Relatedness"),
  n    = c(sum(!is.na(df$LLM_aut)),  sum(!is.na(df$LLM_comp)),  sum(!is.na(df$LLM_rel))),
  mean = round(c(mean(df$LLM_aut, na.rm=TRUE), mean(df$LLM_comp, na.rm=TRUE), mean(df$LLM_rel, na.rm=TRUE)), 3),
  sd   = round(c(sd(df$LLM_aut,   na.rm=TRUE), sd(df$LLM_comp,   na.rm=TRUE), sd(df$LLM_rel,   na.rm=TRUE)), 3),
  min  = round(c(min(df$LLM_aut,  na.rm=TRUE), min(df$LLM_comp,  na.rm=TRUE), min(df$LLM_rel,  na.rm=TRUE)), 3),
  max  = round(c(max(df$LLM_aut,  na.rm=TRUE), max(df$LLM_comp,  na.rm=TRUE), max(df$LLM_rel,  na.rm=TRUE)), 3)
)

cat("LLM descriptives:\n")
print(llm_desc)
save_table(llm_desc, "LLM_descriptives")

llm_long <- data.frame(
  dimension = rep(c("Autonomy", "Competence", "Relatedness"), each = nrow(df)),
  score     = c(df$LLM_aut, df$LLM_comp, df$LLM_rel)
)

p_llm_hist <- ggplot(llm_long, aes(x = score)) +
  geom_histogram(binwidth = 0.25, fill = "coral", alpha = 0.8, color = "white") +
  facet_wrap(~ dimension) +
  scale_x_continuous(breaks = 1:5) +
  labs(title = paste("LLM mean score distributions —", RUN_LABEL),
       x = "LLM mean score", y = "Count") +
  theme_minimal()
save_plot(p_llm_hist, "LLM_mean_histogram", width = 10, height = 4)

p_llm_box <- ggplot(llm_long, aes(x = dimension, y = score, fill = dimension)) +
  geom_boxplot(alpha = 0.7) +
  scale_y_continuous(breaks = 1:5) +
  labs(title = paste("LLM mean score boxplots —", RUN_LABEL),
       x = NULL, y = "LLM mean score") +
  theme_minimal() +
  theme(legend.position = "none")
save_plot(p_llm_box, "LLM_mean_boxplot")

cat(sprintf("\nDone. All outputs saved to %s\n", OUTPUT_DIR))

# ============================================================
# SECTION 4: LLM RELIABILITY — ICC + SD + 95% CI
# ============================================================
cat("\n=== LLM Reliability ===\n\n")

library(irr)

# --- Within-person SD across 3 repetitions per dimension ---
df$sd_aut  <- apply(df[, c("llm_aut_rep1",  "llm_aut_rep2",  "llm_aut_rep3")],  1, sd, na.rm=TRUE)
df$sd_comp <- apply(df[, c("llm_com_rep1", "llm_com_rep2", "llm_com_rep3")], 1, sd, na.rm=TRUE)
df$sd_rel  <- apply(df[, c("llm_rel_rep1",  "llm_rel_rep2",  "llm_rel_rep3")],  1, sd, na.rm=TRUE)

# --- 95% CI for mean LLM score per participant per dimension ---
ci_95 <- function(x) {
  se <- sd(x, na.rm=TRUE) / sqrt(sum(!is.na(x)))
  mean_x <- mean(x, na.rm=TRUE)
  c(lower = mean_x - 1.96 * se, upper = mean_x + 1.96 * se)
}

ci_aut  <- ci_95(df$LLM_aut)
ci_comp <- ci_95(df$LLM_comp)
ci_rel  <- ci_95(df$LLM_rel)

# --- SD summary across participants ---
sd_summary <- data.frame(
  dimension = c("Autonomy", "Competence", "Relatedness"),
  mean_SD   = round(c(mean(df$sd_aut,  na.rm=TRUE),
                      mean(df$sd_comp, na.rm=TRUE),
                      mean(df$sd_rel,  na.rm=TRUE)), 3),
  mean_LLM  = round(c(mean(df$LLM_aut,  na.rm=TRUE),
                      mean(df$LLM_comp, na.rm=TRUE),
                      mean(df$LLM_rel,  na.rm=TRUE)), 3),
  CI_lower  = round(c(ci_aut["lower"],
                      ci_comp["lower"],
                      ci_rel["lower"]), 3),
  CI_upper  = round(c(ci_aut["upper"],
                      ci_comp["upper"],
                      ci_rel["upper"]), 3)
)

cat("Within-person SD and 95% CI for mean LLM score:\n")
print(sd_summary)
save_table(sd_summary, "LLM_reliability_SD_CI")

# --- ICC(2,3): two-way random, absolute agreement, average of 3 measures ---

icc_aut_avg  <- icc(df[, c("llm_aut_rep1", "llm_aut_rep2", "llm_aut_rep3")],
                    model = "twoway", type = "agreement", unit = "average")
icc_comp_avg <- icc(df[, c("llm_com_rep1", "llm_com_rep2", "llm_com_rep3")],
                    model = "twoway", type = "agreement", unit = "average")
icc_rel_avg  <- icc(df[, c("llm_rel_rep1", "llm_rel_rep2", "llm_rel_rep3")],
                    model = "twoway", type = "agreement", unit = "average")

icc_2k_summary <- data.frame(
  dimension = c("Autonomy", "Competence", "Relatedness"),
  ICC_2k    = round(c(icc_aut_avg$value,  icc_comp_avg$value,  icc_rel_avg$value),  3),
  CI_lower  = round(c(icc_aut_avg$lbound, icc_comp_avg$lbound, icc_rel_avg$lbound), 3),
  CI_upper  = round(c(icc_aut_avg$ubound, icc_comp_avg$ubound, icc_rel_avg$ubound), 3)
)

cat("\nICC(2,3) — two-way random, absolute agreement, average of 3 measures:\n")
print(icc_2k_summary)
save_table(icc_2k_summary, "LLM_reliability_ICC2k")




# --- ICC plot ---
p_icc <- ggplot(icc_summary, aes(x = dimension, y = ICC)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), width = 0.15) +
  geom_hline(yintercept = 0.75, linetype = "dashed", color = "gray40") +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25)) +
  labs(title = paste("LLM stability ICC(2,1) —", RUN_LABEL),
       subtitle = "Dashed line = ICC 0.75 reference",
       x = NULL, y = "ICC") +
  theme_minimal()
save_plot(p_icc, "LLM_reliability_ICC_plot")

# --- SD distribution plot per dimension ---
sd_long <- data.frame(
  dimension = rep(c("Autonomy","Competence","Relatedness"), each=nrow(df)),
  sd        = c(df$sd_aut, df$sd_comp, df$sd_rel)
)

p_sd <- ggplot(sd_long, aes(x = sd)) +
  geom_histogram(binwidth = 0.1, fill = "coral", alpha = 0.8, color = "white") +
  facet_wrap(~ dimension) +
  labs(title = paste("Within-person SD across repetitions —", RUN_LABEL),
       x = "SD across 3 repetitions", y = "Count") +
  theme_minimal()
save_plot(p_sd, "LLM_reliability_SD_histogram", width = 10, height = 4)