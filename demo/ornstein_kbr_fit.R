library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
drift = "-x"
diffusion = "sqrt(2)"
x = simulate_sde(drift, diffusion, samplingPeriod = 0.001, tsLength = 10000)
plot.ts(x, ylab = "x(t)", xlab = "Time t", main = "Ornstein–Uhlenbeck process")
# kbr fit  ----------------------------------------------------------
pf = fit_kbr_sde(x, h, driftBw = seq(0.2, 0.4, len = 25),
                 diffBw = seq(0.01, 0.2, len = 25),
                 driftErrorBw = 0.1, diffErrorBw = 0.1,
                 nSim = 700, nthreads = 3)

realDrift = eval(parse(text = drift), list(x = pf$best$x))
plot(pf$best, "drift", ylim = range(c(realDrift, pf$best$drift)))
lines(pf$best$x, realDrift, col = 2)
legend("topright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))
realDiffusion = eval(parse(text = diffusion), list(x = pf$best$x)) ^ 2
if (length(realDiffusion) == 1) {
  realDiffusion = rep(realDiffusion, length(pf$best$x))
}
plot(pf$best, "diff",  ylim = range(c(pf$best$diffusion, realDiffusion)))
lines(pf$best$x, realDiffusion, col = 2)
legend("bottomright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))

