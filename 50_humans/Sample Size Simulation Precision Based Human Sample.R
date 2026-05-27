# ── Sample Size via Accuracy in Parameter Estimation (AIPE) ───────────────────
# Target: Mean 95% CI half-width ≤ 0.20 for Pearson r
# Framework: Estimation-based inference (Amrhein et al., 2019)

# ── Parameters ────────────────────────────────────────────────────────────────
r_true     <- 0.3
n_sims     <- 2000          
n_range    <- seq(50, 300, by = 5)
target_hw  <- 0.15
ci_level   <- 0.95
set.seed(666)

# ── Simulation ────────────────────────────────────────────────────────────────
simulate_ci <- function(n, r, n_sims, ci_level) {
  half_widths <- replicate(n_sims, {
    x  <- rnorm(n)
    y  <- r * x + sqrt(1 - r^2) * rnorm(n)
    ct <- cor.test(x, y, conf.level = ci_level)
    (ct$conf.int[2] - ct$conf.int[1]) / 2
  })
  # Return plain unnamed vector to avoid quantile name collision in sapply
  c(mean(half_widths),
    median(half_widths),
    quantile(half_widths, 0.10, names = FALSE),
    quantile(half_widths, 0.90, names = FALSE))
}

cat("Running simulation — please wait...\n")
results_mat <- sapply(n_range, simulate_ci,
                      r = r_true, n_sims = n_sims, ci_level = ci_level)

# ── Assemble results dataframe ─────────────────────────────────────────────────
results <- data.frame(
  n         = n_range,
  mean_hw   = results_mat[1, ],
  median_hw = results_mat[2, ],
  q10_hw    = results_mat[3, ],
  q90_hw    = results_mat[4, ]
)

# ── Minimum N ─────────────────────────────────────────────────────────────────
min_n <- results$n[which(results$mean_hw <= target_hw)[1]]

cat("=== AIPE Sample Size Results ===\n")
cat(sprintf("True r assumed    : %.2f\n",  r_true))
cat(sprintf("CI level          : %.0f%%\n", ci_level * 100))
cat(sprintf("Target half-width : %.2f\n",  target_hw))
cat(sprintf("Simulations per N : %d\n\n",  n_sims))
cat(sprintf("Minimum N for mean CI half-width <= %.2f : %d\n\n", target_hw, min_n))

# ── Table around threshold ─────────────────────────────────────────────────────
cat("=== Results around threshold ===\n")
window <- results[results$n >= (min_n - 20) & results$n <= (min_n + 20), ]
print(window, digits = 3, row.names = FALSE)

# ── Plot ───────────────────────────────────────────────────────────────────────
plot(results$n, results$mean_hw,
     type = "l", lwd = 2, col = "steelblue",
     xlab = "Sample Size (N)",
     ylab = "Mean 95% CI Half-Width",
     main = sprintf("AIPE Curve  |  r = %.1f,  %d%% CI", r_true, ci_level * 100),
     ylim = c(0, max(results$q90_hw) * 1.05))

polygon(c(results$n, rev(results$n)),
        c(results$q10_hw, rev(results$q90_hw)),
        col = adjustcolor("steelblue", alpha.f = 0.15), border = NA)

abline(h = target_hw, lty = 2, col = "firebrick", lwd = 1.5)
abline(v = min_n,     lty = 2, col = "darkgreen",  lwd = 1.5)

legend("topright",
       legend = c(
         "Mean CI half-width",
         "80% of simulated half-widths",
         sprintf("Target = %.2f", target_hw),
         sprintf("Min N = %d", min_n)
       ),
       col  = c("steelblue", adjustcolor("steelblue", alpha.f = 0.3),
                "firebrick", "darkgreen"),
       lty  = c(1, NA, 2, 2),
       lwd  = c(2, NA, 1.5, 1.5),
       pch  = c(NA, 15, NA, NA),
       pt.cex = 2,
       bty  = "n")