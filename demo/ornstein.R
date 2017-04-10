library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
drift = "-x"
diffusion = "sqrt(2)"
x = simulate_sde(drift, diffusion, samplingPeriod = 0.001, tsLength = 10000)
plot.ts(x, ylab = "x(t)", xlab = "Time t", main = "Ornstein–Uhlenbeck process")

# do inference  ----------------------------------------------------------
m = 10
uncertainty = 5
targetIndex = 1

xm = as.matrix(quantile(x, seq(0.05, 0.95, len = m)), ncol = 1)
diffParams = select_diffusion_parameters(x, h, priorOnSd = uncertainty)
v = diffParams$v
v = -1.5
fk = new(exp_kernel, 1, uncertainty, 1, 1e-5)
sk = new(exp_kernel, 1, diffParams$kernelAmplitude, 1, 1e-5)


inferenceResults = sde_vi(targetIndex, x, h, xm, fk, sk, v, 10, relTol = 1e-3)


# check results -----------------------------------------------------------
plot(inferenceResults$likelihoodLowerBound, main = "Lower Bound")

predictionSupport = matrix(seq(quantile(x,0.05), quantile(x,0.95), len = 100), ncol = 1)
driftPred = predict(inferenceResults$drift, predictionSupport)
diffPred = predict(inferenceResults$diff, predictionSupport, log = TRUE)

realDrift = eval(parse(text = drift), list(x = predictionSupport))
plot(driftPred, main = "Drift", ylim = range(c(realDrift, driftPred$qs)))
lines(predictionSupport, realDrift, col = 2)
realDiff = eval(parse(text = diffusion), list(x = predictionSupport)) ^ 2
if (length(realDiff) == 1) {
  realDiff = rep(realDiff, length(supportX))
}
plot(diffPred, main = "Diffusion", include = TRUE, log = "y",
     ylim = range(c(realDiff, diffPred$qs)))
lines(predictionSupport, realDiff, col = 2)
