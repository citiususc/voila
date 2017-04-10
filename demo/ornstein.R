library("sgpsde")

# simulate Ornstein-Uhlenbeck time series ---------------------------------
h = 0.001
set.seed(100)
drift = "-x"
diffusion = "sqrt(1.5)"
x = simulate_sde(drift, diffusion, samplingPeriod = 0.001, tsLength = 20000)
plot.ts(x, ylab = "x(t)", xlab = "Time t", main = "Ornstein–Uhlenbeck process")

# do inference  ----------------------------------------------------------
m = 10
uncertainty = 5
targetIndex = 1
inputDim = 1
driftLengthScale = 2
diffLengthScale =  2
epsilon = 1e-4
relTol = 1e-6

xm = as.matrix(quantile(x, seq(0.05, 0.95, len = m)), ncol = 1)
diffParams = select_diffusion_parameters(x, h, priorOnSd = uncertainty)
v = diffParams$v
# create exponential kernels through the 'new' function: note that the resulting
# objects are C++ pointers
driftKer = new(exp_kernel, inputDim, uncertainty, driftLengthScale, epsilon)
diffKer = new(exp_kernel, inputDim, diffParams$kernelAmplitude, diffLengthScale, epsilon)

# create linear-exponential kernels using 'sde_kernel' interface. The resulting
# objects hide the use of C++ kernels
driftKer2 = sde_kernel("clamped_exp_lin_kernel",
                       list('maxAmplitude' = uncertainty,
                            'linAmplitude' = uncertainty/ 3 / max((x - median(x)) ^ 2),
                            'linCenter' = median(x),
                            'lengthScales' = driftLengthScale),
                       inputDim, epsilon)

diffKer2 =  sde_kernel("clamped_exp_lin_kernel",
                       list('maxAmplitude' = uncertainty,
                            'linAmplitude' = diffParams$kernelAmplitude/ 3 / max((x - median(x)) ^ 2),
                            'linCenter' = median(x),
                            'lengthScales' = diffLengthScale),
                       inputDim, epsilon)

# perform the inference using the different kernels
expInference = sde_vi(targetIndex, x, h, xm, driftKer, diffKer,
                      v, 10, relTol = relTol)

expLinInference = sde_vi(targetIndex, x, h, xm, driftKer2, diffKer2,
                         v, 10, relTol = relTol)


# check results -----------------------------------------------------------
# check convergence
oldPar = par(mfrow = c(2,1))
plot(expInference$likelihoodLowerBound[-1],
     main = "Lower Bound (exponential kernel)")
plot(expInference$likelihoodLowerBound[-1],
     main = "Lower Bound (exponential-linear kernel)")
par(oldPar)

# get predictions for plotting
predictionSupport = matrix(seq(quantile(x,0.05), quantile(x,0.95), len = 100),
                           ncol = 1)
driftPred = predict(expInference$drift, predictionSupport)
diffPred = predict(expInference$diff, predictionSupport, log = TRUE)

driftPred2 = predict(expLinInference$drift, predictionSupport)
diffPred2 = predict(expLinInference$diff, predictionSupport, log = TRUE)

# plot drift
realDrift = eval(parse(text = drift), list(x = predictionSupport))
plot(predictionSupport, realDrift,
     ylim = range(c(realDrift, driftPred$qs, driftPred2$qs)), type = "l",
     main = "Drift")
lines(driftPred, col = 2)
lines(driftPred2, col = 3, lty = 3)
legend("topright", lty = 1:3, col = 1:3,
       legend = c("Real", "Exp estimate", "Lin-exp estimate"), bty = "n")

# plot diff
realDiff = eval(parse(text = diffusion), list(x = predictionSupport)) ^ 2
if (length(realDiff) == 1) {
  realDiff = rep(realDiff, length(supportX))
}
plot(predictionSupport, realDiff,
     ylim = range(c(realDiff, diffPred$qs, diffPred2$qs)) * c(1, 1.05),
     type = "l", main = "Diffusion")
lines(diffPred, col = 2, lty = 2)
lines(diffPred2, col = 3, lty = 3)
legend("topright", lty = 1:3, col = 1:3,
       legend = c("Real", "Exp estimate", "Lin-exp estimate"), bty = "n")


print(expLinInference$drift$gpKernel)
print(expLinInference$diff$gpKernel)
beepr::beep()
