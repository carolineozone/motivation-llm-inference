# 00_common.R — shared constants, palettes, themes, and helpers
# Source this file at the top of every analysis QMD's setup chunk.
# All packages must be loaded by the sourcing QMD before source() is called.
# ─────────────────────────────────────────────────────────────────────────────

# ── 1  STATISTICAL CONSTANTS ─────────────────────────────────────────────────

BOOTSTRAP_SAMPLES <- 10000L          # primary bootstrap R (RQ2a, RQ2b, RQ3)
POLY_BOOT_R_RQ2   <- 1000L           # Spearman + Polychoric bootstrap R (slower)

CI_LEVEL     <- .95
CI_BOUND_95  <- 1.96
CI_Q_LO      <- (1 - CI_LEVEL) / 2        # 0.025
CI_Q_HI      <- 1 - (1 - CI_LEVEL) / 2   # 0.975
CI_LABEL     <- paste0(CI_LEVEL * 100, "% CI")

# ── 2  STUDY DESIGN ──────────────────────────────────────────────────────────

WORD_LENGTH_VARIANTS <- c("50", "150", "300")
WORD_LENGTH_LABELS   <- c("50" = "short", "150" = "medium", "300" = "long")
N_TEXTS_VARIANTS     <- c(1L, 3L, 5L)
N_REPEATS            <- c(1L, 2L, 3L)

CONDITIONS <- paste0(
  rep(WORD_LENGTH_LABELS[WORD_LENGTH_VARIANTS], each = length(N_TEXTS_VARIANTS)),
  "-",
  rep(N_TEXTS_VARIANTS, times = length(WORD_LENGTH_LABELS))
)

ROWS_PER_PERSONA     <- length(CONDITIONS) * length(N_REPEATS)  # 27
TEXTS_PER_PERSONA    <- length(WORD_LENGTH_VARIANTS) * max(N_TEXTS_VARIANTS)  # 15
TEXT_TRUNC_THRESHOLD <- 0.75
TARGET_SAMPLE        <- 400L
TOTAL_EXPECTED_ROWS  <- TARGET_SAMPLE * ROWS_PER_PERSONA  # 10800

# ── 3  TYPOGRAPHY ────────────────────────────────────────────────────────────

THESIS_FONT     <- "Calibri"
THESIS_FONTSIZE <- 11

# ── 4  VISUALIZATION CONSTANTS ───────────────────────────────────────────────

# requires ggplot2
DODGE <- position_dodge(width = 30)

COLOR_ERRORBAR    <- "grey"
COLOR_LEGEND_FRAME <- "grey"
COL_GRID          <- "#E6E6E6"
COL_LBL_INSIDE    <- "#FFFFFF"
COL_LBL_OUTSIDE   <- "#B3B3B3"

# ── 5  COLOR PALETTES ────────────────────────────────────────────────────────

# Per-need palettes — AI (blue) and Human (red)
COLOR_AI <- c(Autonomy = "#084594", Competence = "#4292c6", Relatedness = "#9ecae1")
COLOR_HU <- c(Autonomy = "#800026", Competence = "#e31a1c", Relatedness = "#fcbba1")
COLORS   <- c(COLOR_HU, COLOR_AI)      # combined for scale_fill_manual

# Single-hue anchors
COL_AI_SINGLE      <- "#1F5A71"
COL_HU_SINGLE      <- "#BE2A16"
COL_NEUTRAL_SINGLE <- "#B3B3B3"
COL_DIFF           <- "#DDA613"        # yellow — used for difference annotations

# Shade sequences (index 1 = darkest, 5 = lightest, 6 = white)
COL_AI_SHADES <- c("#1F5A71", "#2085AB", "#489CC1", "#6DBCE0", "#9ecae1", "#FFFFFF")
COL_HU_SHADES <- c("#BE2A16", "#CE5545", "#DC7D71", "#E9A39B", "#F5C9C5", "#FFFFFF")

# Blue-gray Q shades (used alongside COL_AI_SHADES for LLM vs Q overlay)
COL_AI_SHADES2 <- c("#264D6B", "#3B7DA0", "#61AACF", "#91C9E4", "#C0E2F2", "#FFFFFF")

# Convenience aliases
COL_AI_LINE <- COL_AI_SHADES[1]
COL_HU_LINE <- COL_HU_SHADES[1]

# Condition-mapped shades
COL_AI_TEXTS <- c("1" = COL_AI_SHADES[4], "3" = COL_AI_SHADES[2], "5" = COL_AI_SHADES[1])
COL_AI_WORDS <- c("50" = COL_AI_SHADES[4], "150" = COL_AI_SHADES[2], "300" = COL_AI_SHADES[1])

# Per-need palettes: LLM scores (darker) vs Q scores (blue-gray)
COL_AI_LLM <- c(Autonomy = COL_AI_SHADES[1],  Competence = COL_AI_SHADES[2],  Relatedness = COL_AI_SHADES[4])
COL_AI_Q   <- c(Autonomy = COL_AI_SHADES2[1], Competence = COL_AI_SHADES2[2], Relatedness = COL_AI_SHADES2[3])

# ── 6  NEED AND SAMPLE LABELS ────────────────────────────────────────────────

AUT <- "autonomy"
COM <- "competence"
REL <- "relatedness"

TABLE_LABEL_AUT <- "Autonomy"
TABLE_LABEL_COM <- "Competence"
TABLE_LABEL_REL <- "Relatedness"
NEEDS <- c(TABLE_LABEL_AUT, TABLE_LABEL_COM, TABLE_LABEL_REL)
ITEMS <- paste0(rep(NEEDS, each = 4), " ", 1:4)

LABEL_SAMPLE <- "Sample"
LABEL_AI     <- "AI surrogates"
LABEL_HU     <- "Humans"

LABEL_WORDC   <- "Word Count (LLM)"
LABEL_NEEDS   <- "Need Dimension"
LABEL_LLM_AUT <- paste("LLM", TABLE_LABEL_AUT)
LABEL_LLM_COM <- paste("LLM", TABLE_LABEL_COM)
LABEL_LLM_REL <- paste("LLM", TABLE_LABEL_REL)
LABEL_Q_AUT   <- paste("Questionnaire", TABLE_LABEL_AUT)
LABEL_Q_COM   <- paste("Questionnaire", TABLE_LABEL_COM)
LABEL_Q_REL   <- paste("Questionnaire", TABLE_LABEL_REL)

# ── 7  GGPLOT2 THEME ─────────────────────────────────────────────────────────

# requires ggplot2
theme_thesis <- function() {
  ggplot2::theme_minimal(base_family = THESIS_FONT, base_size = THESIS_FONTSIZE) +
    ggplot2::theme(
      strip.text         = ggplot2::element_text(face = "plain", colour = "black",
                                                  size = THESIS_FONTSIZE),
      panel.grid.minor   = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      legend.position    = "top",
      axis.title.x       = ggplot2::element_text(margin = ggplot2::margin(t = 10)),
      axis.title.y       = ggplot2::element_text(margin = ggplot2::margin(r = 10))
    )
}

# ── 8  FLEXTABLE HELPERS ─────────────────────────────────────────────────────
# requires flextable + officer

# APA 7 table style: Calibri 11 pt, full triple-line border set, top-aligned body
style_apa7 <- function(ft) {
  ft |>
    flextable::font(fontname = THESIS_FONT, part = "all") |>
    flextable::fontsize(size = THESIS_FONTSIZE, part = "all") |>
    flextable::border_remove() |>
    flextable::hline_top(part = "header",
                         border = officer::fp_border(width = 1.5)) |>
    flextable::hline_bottom(part = "header",
                            border = officer::fp_border(width = 1)) |>
    flextable::hline_bottom(part = "body",
                            border = officer::fp_border(width = 1.5)) |>
    flextable::align(align = "left", part = "all") |>
    flextable::valign(valign = "top", part = "body") |>
    flextable::padding(padding = 3, part = "all") |>
    flextable::set_table_properties(layout = "autofit")
}

# APA 7 "Note." footer line
add_apa_note <- function(ft, note_text) {
  ft |>
    flextable::add_footer_lines("") |>
    flextable::mk_par(
      part = "footer", i = 1, j = 1,
      value = flextable::as_paragraph(
        flextable::as_chunk(
          "Note. ",
          props = officer::fp_text(italic = TRUE,
                                           font.family = THESIS_FONT,
                                           font.size   = THESIS_FONTSIZE)
        ),
        flextable::as_chunk(
          note_text,
          props = officer::fp_text(italic = FALSE,
                                           font.family = THESIS_FONT,
                                           font.size   = THESIS_FONTSIZE)
        )
      )
    )
}

# ── 9  STEIGER TEST HELPERS ──────────────────────────────────────────────────
# requires cocor

# Pearson-based Steiger test for one condition
# Returns a 1-row data.frame with r_within, r_cross, diff, 95% CI, z, p, n
run_steiger_cond <- function(data, llm_var, q_target, q_other,
                             label_target, label_other, cond_label) {
  x  <- data[[llm_var]]
  y1 <- data[[q_target]]
  y2 <- data[[q_other]]

  complete <- complete.cases(x, y1, y2)
  x  <- x[complete]; y1 <- y1[complete]; y2 <- y2[complete]
  n  <- length(x)

  r_within <- cor(x, y1)
  r_cross  <- cor(x, y2)
  r_y1y2   <- cor(y1, y2)

  result <- cocor::cocor.dep.groups.overlap(
    r.jk = r_within, r.jh = r_cross, r.kh = r_y1y2,
    n = n, alternative = "two.sided", test = "steiger1980"
  )

  diff   <- r_within - r_cross
  z1     <- 0.5 * log((1 + r_within) / (1 - r_within))
  z2     <- 0.5 * log((1 + r_cross)  / (1 - r_cross))
  z_stat <- result@steiger1980$statistic
  # SE back-computed from cocor's z: consistent with the dependent-groups test
  se_eff <- if (abs(z_stat) > 1e-10) abs(z1 - z2) / abs(z_stat) else sqrt(2 / (n - 3))
  crit   <- qnorm(0.975)

  data.frame(
    condition  = cond_label,
    llm_dim    = llm_var,
    comparison = paste(label_target, "vs", label_other),
    r_within   = round(r_within, 2),
    r_cross    = round(r_cross,  3),
    diff       = round(diff,     3),
    ci_lo      = round(tanh((z1 - z2) - crit * se_eff), 2),
    ci_hi      = round(tanh((z1 - z2) + crit * se_eff), 2),
    steiger_z  = round(result@steiger1980$statistic, 2),
    p_value    = round(result@steiger1980$p.value,   3),
    n          = n
  )
}

# Spearman-based Steiger (approximation — note in prose)
run_steiger_cond_spearman <- function(data, llm_var, q_target, q_other,
                                      label_target, label_other, cond_label) {
  x  <- data[[llm_var]]
  y1 <- data[[q_target]]
  y2 <- data[[q_other]]

  complete <- complete.cases(x, y1, y2)
  x  <- x[complete]; y1 <- y1[complete]; y2 <- y2[complete]
  n  <- length(x)

  r_within <- cor(x, y1, method = "spearman")
  r_cross  <- cor(x, y2, method = "spearman")
  r_y1y2   <- cor(y1, y2, method = "spearman")

  result <- cocor::cocor.dep.groups.overlap(
    r.jk = r_within, r.jh = r_cross, r.kh = r_y1y2,
    n = n, alternative = "two.sided", test = "steiger1980"
  )

  diff   <- r_within - r_cross
  z1     <- 0.5 * log((1 + r_within) / (1 - r_within))
  z2     <- 0.5 * log((1 + r_cross)  / (1 - r_cross))
  z_stat <- result@steiger1980$statistic
  # SE back-computed from cocor's z: consistent with the dependent-groups test
  se_eff <- if (abs(z_stat) > 1e-10) abs(z1 - z2) / abs(z_stat) else sqrt(2 / (n - 3))
  crit   <- qnorm(0.975)

  data.frame(
    condition  = cond_label,
    llm_dim    = llm_var,
    comparison = paste(label_target, "vs", label_other),
    rho_within = round(r_within, 2),
    rho_cross  = round(r_cross,  3),
    diff       = round(diff,     3),
    ci_lo      = round(tanh((z1 - z2) - crit * se_eff), 2),
    ci_hi      = round(tanh((z1 - z2) + crit * se_eff), 2),
    steiger_z  = round(result@steiger1980$statistic, 2),
    p_value    = round(result@steiger1980$p.value,   3),
    n          = n
  )
}

message("00_common.R loaded.")
