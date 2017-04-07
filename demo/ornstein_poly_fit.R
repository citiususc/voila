library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
#set.seed(1)
model = suppressWarnings(setModel(drift = "-x",
                                  diffusion = "sqrt(2)"))
X = suppressWarnings(simulate(model,
                              sampling = setSampling(delta = h, n = 20000)))
x = as.matrix(get.zoo.data(X)[[1]], ncol = 1)
plot.ts(x)
# poly fit  ----------------------------------------------------------
pf = fit_polynomial_sde(x, h, nDrift = 3, nDiff = 5)

supportX = seq(min(x), max(x), len = 100)
f = predict(pf$drift, supportX)
plot(supportX, f, type = "l", ylim = range(c(-supportX, f)))
lines(supportX, -supportX, col = 2)
g2 = predict(pf$diff, supportX)
plot(supportX, g2, type = "l")
lines(supportX, rep(2, length(supportX)), col = 2)

