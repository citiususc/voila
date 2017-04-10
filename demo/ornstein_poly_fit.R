library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
#set.seed(1)
drift = "-x"
diffusion = "sqrt(2)"
x = simulate_sde(drift, diffusion, samplingPeriod = 0.001, tsLength = 10000)
plot.ts(x, ylab = "x(t)", xlab = "Time t", main = "Ornstein–Uhlenbeck process")
# poly fit  ----------------------------------------------------------
pf = fit_polynomial_sde(x, h, nDrift = 3, nDiff = 5)

supportX = seq(min(x), max(x), len = 100)
f = predict(pf$drift, supportX)
realDrift = eval(parse(text = drift), list(x = supportX))
plot(supportX, f, type = "l", ylim = range(c(realDrift, f)))
lines(supportX, realDrift, col = 2)
legend("topright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))

g2 = predict(pf$diff, supportX)
realDiffusion = eval(parse(text = diffusion), list(x = supportX)) ^ 2
if (length(realDiffusion) == 1) {
  realDiffusion = rep(realDiffusion, length(supportX))
}
plot(supportX, g2, type = "l", ylim = range(c(realDiffusion, g2)))
lines(supportX, realDiffusion, col = 2)
legend("bottomright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))
