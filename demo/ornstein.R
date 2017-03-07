library("yuima")
h = 0.001
set.seed(1)
model = suppressWarnings(setModel(drift = "-x",
                                  diffusion = "sqrt(2)"))
X = suppressWarnings(simulate(model,
                              sampling = setSampling(delta = h, n = 5000)))
x = as.matrix(get.zoo.data(X)[[1]], ncol = 1)
m = 5
xm = as.matrix(quantile(x, seq(0.05, 0.95, len = m)), ncol = 1)
targ = as.numeric(diff(x))

fk = new(exp_kernel, 1, 5, 1, 1e-5)
sk = new(exp_kernel, 1, 5, 1, 1e-5)
feat = x[-nrow(x),,drop = FALSE]
v = -1.5
tmp = microbenchmark::microbenchmark(
  pars <- sde_vi(0, x , h, xm, fk, sk, v, 100, -1),
  times = 5)
print(tmp)
print(pars)

plot(pars$Ls,log="y")
