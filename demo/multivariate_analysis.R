library('voila')
# driftExpression = c("-x1 * x2","2 - x2")
# diffExpression =  matrix( c(sqrt(1.5), 0,
#                             0, "0.1 * x2"), 2, 2)

set.seed(100)
driftExpression = c("x2", "-2.5 ^ 2 * x1")
diffExpression = matrix(c("sqrt(2)", 0,
                          0, "sqrt(2 + x1 ^ 2 + 0.5 * x2 ^ 2)"), 2, 2)
h = 0.001
x  = simulate_sde(driftExpression,
             diffExpression,
             h, 10000,
             stateVariable = c("x1", "x2"))

oldPar = par(mfrow = c(3,1), mar = c(2,2,2,0.25))
plot.ts(x[,1])
plot.ts(x[,2])
plot(x)
#  ------------------------------------------------------------------------
epsilon = 1e-4
relTol = 1e-6
uncertainty = 10
inputDim = 2
timeSeriesIdx = 2

debugonce(sde_kernel)
sde_kernel('exp_kernel',list('amplitude'=5, 'lengthScales'=c(2,2)),2)

diffParams = select_diffusion_parameters(x, h, priorOnSd = uncertainty,
                                         responseVariableIndex = timeSeriesIdx)
driftKer = new(exp_kernel, inputDim, uncertainty, c(2,2), epsilon)
diffKer = new(exp_kernel, inputDim, diffParams$kernelAmplitude, c(2,2), epsilon)

library('mvtnorm')
xm = rmvnorm(10, colMeans(x), cov(x))
inference = sde_vi(timeSeriesIndex = 2, x, h, xm, driftKer, diffKer, diffParams$v,
                   maxIterations = 30)


# Plot the estimates ------------------------------------------------------

x1 = seq(quantile(x[,1], 0.05), quantile(x[,1], 0.95), len = 100)
x2 = seq(quantile(x[,2], 0.05), quantile(x[,2], 0.95), len = 100)

library('plot3D')
par(mfrow = c(2,1), mar = c(0,2,2,0))

zDrift = outer(x1, x2, function(x1, x2) {
  eval(parse(text = driftExpression[[timeSeriesIdx]]),
       list(x1 = x1, x2 = x2))

})
zDriftEst = outer(x1, x2, function(x1, x2) predict(inference$drift,
                                                  matrix(c(x1,x2), ncol = 2),
                                                  lognormal = FALSE)$mean
)
col = seq(min(c(zDrift, zDriftEst)), max(c(zDrift, zDriftEst)), len = 25)
persp3D(x = x1,y = x2, zDrift, phi = 35, ticktype = 'detailed',
        zlim = range(c(zDriftEst, zDrift)), breaks = col)
persp3D(x = x1,y = x2, zDriftEst, phi = 35, ticktype = 'detailed',
        zlim = range(c(zDriftEst, zDrift)), breaks = col)
scatter3D(inference$inducingPoints[,1], inference$inducingPoints[,2],
          apply(inference$inducingPoints, 1, function(x){
            predict(inference$drift, matrix(x, ncol = 2))$mean
          }), col = 'black', bg = 'black', pch = 21, add = TRUE)


zDiff = outer(x1, x2, function(x1, x2) {
  eval(parse(text = diffExpression[timeSeriesIdx,timeSeriesIdx]),
       list(x1 = x1, x2 = x2)) ^ 2 + rnorm(length(x1), sd = 0.001)

})
zDiffEst = outer(x1, x2, function(x1, x2) predict(inference$diff,
                                                  matrix(c(x1,x2), ncol = 2),
                                                  lognormal = TRUE)$mean
)
col = seq(min(c(zDiff, zDiffEst)), max(c(zDiff, zDiffEst)), len = 25)
persp3D(x = x1,y = x2, zDiff, phi = 35, ticktype = 'detailed',
        zlim = range(c(zDiff, zDiffEst)), breaks = col)
persp3D(x = x1,y = x2, zDiffEst, phi = 35, ticktype = 'detailed',
        zlim = range(c(zDiff, zDiffEst)), breaks = col)
scatter3D(inference$inducingPoints[,1], inference$inducingPoints[,2],
          apply(inference$inducingPoints, 1, function(x){
            predict(inference$diff, matrix(x, ncol = 2), lognormal = TRUE)$mean
          }), col = 'black', bg = 'black', pch = 21, add = TRUE)

