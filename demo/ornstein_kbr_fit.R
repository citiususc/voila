library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
set.seed(1)
model = suppressWarnings(setModel(drift = "-x",
                                  diffusion = "sqrt(2)"))
X = suppressWarnings(simulate(model,
                              sampling = setSampling(delta = h, n = 10000)))
x = as.matrix(get.zoo.data(X)[[1]], ncol = 1)
plot.ts(x, ylab = "x(t)", xlab = "Time t", main = "Ornstein–Uhlenbeck process")
# kbr fit  ----------------------------------------------------------
pf = fit_kbr_sde(x, h, driftBw = seq(0.8, 1.5, len = 50),
                 diffBw = seq(0.5, 1.5, len = 50),
                 nSim = 1000, nthreads = 3)

plot(pf$best, "drift")
lines(pf$best$x, -pf$best$x, col = 2)
legend("topright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))
plot(pf$best, "diff")
lines(pf$best$x, rep(2, length(pf$best$x)), col = 2)
legend("bottomright", col = 1:2, lty = 1, legend = c("Estimate", "Real"))

