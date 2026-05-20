# Script: Exclusions_questionnaire.R
# Applies preregistered exclusion criteria to questionnaire data
# Criteria: long-string >= 9 consecutive identical responses,
#           within-person SD < 0.20 across all Likert items
#           missing data flagged per subscale
# References: DeSimone et al. (2015), Huang et al. (2012), Meade & Craig (2012)
#
# Input:  Data_Complete_Run1.csv
# Output: flagged dataset + summary saved to OUTPUT_DIR

library(dplyr)

# --- Configuration ---
INPUT_FILE           <- "Data_Complete_Run2.csv"
OUTPUT_FILE          <- "Data_Complete_Run2_flagged.csv"
RUN_LABEL            <- "Run1_Son46_T03_PrD"
OUTPUT_DIR           <- "/Users/seb/Library/CloudStorage/Dropbox/Akademin/Master Psykology/Master thesis project course/Thesis/Code/R-MTSebastian/60_Plots_Sebastian"
LONGSTRING_THRESHOLD <- 9
SD_THRESHOLD         <- 0.20

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

save_table <- function(df, name) {
  filename <- file.path(OUTPUT_DIR, paste0(RUN_LABEL, "_", name, ".csv"))
  write.table(df, filename, sep = "|", row.names = FALSE, quote = FALSE)
  cat(sprintf("  Table saved: %s\n", basename(filename)))
}

# --- Load data ---
df <- read.table(INPUT_FILE, header = TRUE, sep = "|",
                 stringsAsFactors = FALSE)
cat("Participants loaded:", nrow(df), "\n")

# --- Define item columns ---
pvq_cols  <- names(df)[grepl("^PVQ_", names(df))]
bpns_cols <- c("aut1","aut2","aut3","aut4",
               "comp1","comp2","comp3","comp4",
               "rel1","rel2","rel3","rel4","rel5")
all_likert <- c(pvq_cols, bpns_cols)

cat("PVQ columns:", length(pvq_cols), "\n")
cat("BPNS columns:", length(bpns_cols), "\n")
cat("Total Likert items:", length(all_likert), "\n\n")

# --- Function: longest consecutive run of identical values ---
longest_run <- function(x) {
  x <- suppressWarnings(as.numeric(unlist(x)))
  x <- x[!is.na(x)]
  if (length(x) == 0) return(0)
  max(rle(x)$lengths)
}

# --- FLAG 1: Long-string ---
df$longstring_length <- apply(
  df[, all_likert], 1,
  function(row) longest_run(row)
)
df$flag_longstring <- df$longstring_length >= LONGSTRING_THRESHOLD

# --- FLAG 2: Within-person SD ---
df$within_person_sd <- apply(
  df[, all_likert], 1,
  function(row) round(sd(suppressWarnings(as.numeric(row)), na.rm = TRUE), 3)
)
df$flag_low_sd <- df$within_person_sd < SD_THRESHOLD

# --- FLAG 3: Missing data per subscale ---
aut_items  <- c("aut1","aut2","aut3","aut4")
comp_items <- c("comp1","comp2","comp3","comp4")
rel_items  <- c("rel1","rel2","rel3","rel4","rel5")

df$missing_aut  <- rowSums(is.na(df[, aut_items]))
df$missing_comp <- rowSums(is.na(df[, comp_items]))
df$missing_rel  <- rowSums(is.na(df[, rel_items]))

df$pct_missing_aut  <- df$missing_aut  / length(aut_items)
df$pct_missing_comp <- df$missing_comp / length(comp_items)
df$pct_missing_rel  <- df$missing_rel  / length(rel_items)

# Flag if >20% missing on any subscale
df$flag_missing <- (df$pct_missing_aut  > 0.20 |
                      df$pct_missing_comp > 0.20 |
                      df$pct_missing_rel  > 0.20)

# --- Combined exclusion flag ---
df$flag_exclude <- df$flag_longstring | df$flag_low_sd | df$flag_missing

# --- Summary ---
cat("=== Exclusion Flag Summary ===\n\n")

cat(sprintf("Long-string (>= %d consecutive identical responses):\n",
            LONGSTRING_THRESHOLD))
cat(sprintf("  Flagged: %d of %d participants\n\n",
            sum(df$flag_longstring), nrow(df)))

cat(sprintf("Low within-person SD (< %.2f):\n", SD_THRESHOLD))
cat(sprintf("  Flagged: %d of %d participants\n\n",
            sum(df$flag_low_sd), nrow(df)))

cat("Missing data (>20% on any BPNS subscale):\n")
cat(sprintf("  Flagged: %d of %d participants\n\n",
            sum(df$flag_missing), nrow(df)))

cat("Combined (flagged on any criterion):\n")
cat(sprintf("  Flagged: %d of %d participants\n",
            sum(df$flag_exclude), nrow(df)))
cat(sprintf("  Remaining after exclusion: %d\n\n",
            sum(!df$flag_exclude)))

# --- Inspect flagged cases ---
flagged <- df %>%
  filter(flag_exclude) %>%
  select(participant_id, longstring_length, within_person_sd,
         pct_missing_aut, pct_missing_comp, pct_missing_rel,
         flag_longstring, flag_low_sd, flag_missing, flag_exclude)

if (nrow(flagged) > 0) {
  cat("=== Flagged Participants ===\n")
  print(flagged)
  save_table(flagged, "exclusions_flagged_participants")
} else {
  cat("No participants flagged.\n")
}

# --- Save full flagged dataset ---
write.table(df, OUTPUT_FILE, row.names = FALSE, sep = "|", quote = FALSE)
cat(sprintf("\nFull flagged dataset saved to %s\n", OUTPUT_FILE))

# --- Save exclusion summary ---
excl_summary <- data.frame(
  criterion    = c("Long-string", "Low SD", "Missing data", "Combined"),
  n_flagged    = c(sum(df$flag_longstring), sum(df$flag_low_sd),
                   sum(df$flag_missing),    sum(df$flag_exclude)),
  n_remaining  = c(nrow(df) - sum(df$flag_longstring),
                   nrow(df) - sum(df$flag_low_sd),
                   nrow(df) - sum(df$flag_missing),
                   sum(!df$flag_exclude))
)
print(excl_summary)

# --- Full exclusion detail table ---
excl_detail <- df %>%
  select(participant_id, 
         longstring_length, within_person_sd,
         missing_aut, missing_comp, missing_rel,
         pct_missing_aut, pct_missing_comp, pct_missing_rel,
         flag_longstring, flag_low_sd, flag_missing, flag_exclude)

save_table(excl_detail, "exclusions_full_detail")
cat(sprintf("Full exclusion detail table saved (%d participants)\n", nrow(excl_detail)))

save_table(excl_summary, "exclusions_summary")