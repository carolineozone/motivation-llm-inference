# Sample size (N) Simulation for Pearson Correlation
# Two-tailed hypothesis, alpha = 0.05, target power = 0.90
# True effect: r = 0.3

# ── Parameters ────────────────────────────────────────────────────────────────
r_true    <- 0.3
alpha     <- 0.05
target    <- 0.90
n_sims    <- 10000
n_range   <- seq(50, 200, by = 5)
set.seed(666)

# ── Simulation ────────────────────────────────────────────────────────────────
simulate_power <- function(n, r, n_sims, alpha) {
  sig <- replicate(n_sims, {
    # Bivariate normal with correlation r
    x <- rnorm(n)
    y <- r * x + sqrt(1 - r^2) * rnorm(n)
    cor.test(x, y, alternative = "two.sided")$p.value < alpha
  })
  mean(sig)
}

power_results <- sapply(n_range, simulate_power,
                        r = r_true, n_sims = n_sims, alpha = alpha)

# ── Minimum N ─────────────────────────────────────────────────────────────────
min_n <- n_range[which(power_results >= target)[1]]

cat("=== Power Simulation Results ===\n")
cat(sprintf("True r       : %.2f\n", r_true))
cat(sprintf("Alpha        : %.3f (two-tailed)\n", alpha))
cat(sprintf("Target power : %.2f\n", target))
cat(sprintf("Simulations  : %d per N\n\n", n_sims))
cat(sprintf("Minimum N for %.0f%% power: %d\n\n", target * 100, min_n))

# ── Analytical check (pwr package) ────────────────────────────────────────────
if (requireNamespace("pwr", quietly = TRUE)) {
  library(pwr)
  analytic <- pwr.r.test(r = r_true, sig.level = alpha, power = target,
                         alternative = "two.sided")
  cat(sprintf("Analytical N (pwr package): %d\n", ceiling(analytic$n)))
} else {
  cat("Install the 'pwr' package for an analytical comparison: install.packages('pwr')\n")
}

# ── Plot ──────────────────────────────────────────────────────────────────────
plot(n_range, power_results,
     type = "l", lwd = 2, col = "steelblue",
     xlab = "Sample Size (N)",
     ylab = "Estimated Power",
     main = sprintf("Power Curve  |  r = %.1f, alpha = %.2f, two-tailed", r_true, alpha),
     ylim = c(0, 1))
abline(h  = target, lty = 2, col = "firebrick", lwd = 1.5)
abline(v  = min_n,  lty = 2, col = "darkgreen",  lwd = 1.5)
legend("bottomright",
       legend = c("Simulated power",
                  sprintf("Target = %.2f", target),
                  sprintf("Min N = %d", min_n)),
       col    = c("steelblue", "firebrick", "darkgreen"),
       lty    = c(1, 2, 2), lwd = 2, bty = "n")

