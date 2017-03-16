library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
#set.seed(1)
model = suppressWarnings(setModel(drift = "-x",
                                  diffusion = "sqrt(2)"))
X = suppressWarnings(simulate(model,
                              sampling = setSampling(delta = h, n = 10000)))
x = as.matrix(get.zoo.data(X)[[1]], ncol = 1)

# do inference  ----------------------------------------------------------
m = 10
uncertainty = 5
targetIndex = 1

xm = as.matrix(quantile(x, seq(0.05, 0.95, len = m)), ncol = 1)
diffParams = select_diffusion_parameters(x, h, priorOnSd = uncertainty)
v = diffParams$v
v = -1.5
fk = sde_kernel("exp_kernel", list('amplitude' = uncertainty, 'lengthScales' = 1),
                 1, 1e-5)
sk = sde_kernel("exp_kernel", list('amplitude' = diffParams$kernelAmplitude,
                                   'lengthScales' = 1), 1, 1e-5)


inferenceResults = sde_vi(targetIndex, x, h, xm, fk, sk, v, 4, relTol = 1e-3)


# check results -----------------------------------------------------------
plot(inferenceResults$likelihoodLowerBound, main = "Lower Bound")

predictionSupport = matrix(seq(quantile(x,0.05), quantile(x,0.95), len = 100), ncol = 1)
driftPred = predict(inferenceResults$drift, predictionSupport)
diffPred = predict(inferenceResults$diff, predictionSupport, log = TRUE)

plot(driftPred, main = "Drift")
lines(predictionSupport, -predictionSupport, col = 2)
plot(diffPred, main = "Diffusion", include = TRUE, log = "y")
lines(predictionSupport, rep(2, nrow(predictionSupport)), col = 2)