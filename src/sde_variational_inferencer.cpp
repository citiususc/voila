#define ARMA_DONT_PRINT_ERRORS
#include "sde_variational_inferencer.h"
#include "utils.h"
#include <cassert>

sde_variational_inferencer::sde_variational_inferencer(kernel& fKernel,
                                                       kernel& sKernel,
                                                       double v)
  :  problem(),
     mFKernel((assert(fKernel.get_input_dimension() == sKernel.get_input_dimension()),
               fKernel)),
               mSKernel(sKernel), mV(v),
               mNoFHyperparameters(fKernel.get_hyperparams().n_elem),
               mNoSHyperparameters(sKernel.get_hyperparams().n_elem),
               mLaplaceSolver(),
               mLowerBoundSolver() {
  mLaplaceSolver.set_projected_gradient_tolerance(0);
  mLowerBoundSolver.set_projected_gradient_tolerance(0);
  mLowerBoundSolver.set_max_iterations(5);
}

sde_variational_inferencer::sde_variational_inferencer(kernel& fKernel,
                                                       kernel& sKernel,
                                                       double v,
                                                       l_bfgs_b<arma::vec>& laplaceSolver,
                                                       l_bfgs_b<arma::vec>& lowerBoundSolver)
  : problem(),
    mFKernel((assert(fKernel.get_input_dimension() == sKernel.get_input_dimension()),
              fKernel)),
              mSKernel(sKernel), mV(v),
              mNoFHyperparameters(fKernel.get_hyperparams().n_elem),
              mNoSHyperparameters(sKernel.get_hyperparams().n_elem),
              mLaplaceSolver(laplaceSolver),
              mLowerBoundSolver(lowerBoundSolver) {
}

int sde_variational_inferencer::get_max_iterations() const {
  return mMaxIt;
}

void sde_variational_inferencer::set_max_iterations(int maxIterations) {
  if (maxIterations <= 0) {
    throw std::invalid_argument("The maximum number of iterations should be > 0");
  }
  mMaxIt = maxIterations;
}

bool sde_variational_inferencer::get_verbose() const {
  return mVerbose;
}

void sde_variational_inferencer::set_verbose(bool verbose) {
  mVerbose = verbose;
}

double sde_variational_inferencer::get_rel_tolerance() const {
  return mRelativeTolerance;
}

void sde_variational_inferencer::set_rel_tolerance(double relTolerance) {
  if (relTolerance < 0) {
    throw std::invalid_argument("Relative tolerance should be >= 0");
  }
  mRelativeTolerance = relTolerance;
}

Rcpp::List sde_variational_inferencer::do_inference(const arma::mat& timeSeries,
                                                    double samplingPeriod,
                                                    arma::mat& xm,
                                                    int targetIndex){
  if (timeSeries.n_cols != mFKernel.get_input_dimension()) {
    throw std::invalid_argument("The time series' dimension does not match \
                                 the kernels' input dimension");
  }
  if (timeSeries.n_cols != xm.n_cols) {
    throw std::invalid_argument("The time series' dimension does not match \
                                 the inducing points' dimension");
  }
  if (xm.n_rows < 2) {
    throw std::invalid_argument("The number of inducing-points should be >= 2");
  }
  if (samplingPeriod < 0) {
    throw std::invalid_argument("Invalid sampling period (< 0)");
  }
  if (targetIndex < 0 || targetIndex > (timeSeries.n_cols - 1)) {
    throw std::invalid_argument("targetIndex out of bounds");
  }

  mHeadX = timeSeries.submat(0, 0, timeSeries.n_rows - 2, timeSeries.n_cols - 1);
  mSamplingPeriod = samplingPeriod;
  mDX = arma::diff(timeSeries, 1);
  mTarget = mDX.col(targetIndex);
  mDXSize = mDX.n_rows;
  mNoPseudoInputs = xm.n_rows;
  mXm = xm;

  Rcpp::Rcout << "Let's do some VI" << std::endl;
  mFKernel.greet();

  // set the variables related to the class problem properly
  // TODO: create a protected set_input_dimension that resets also
  // the bounds
  mInputDimension = mNoFHyperparameters + mNoSHyperparameters + mNoPseudoInputs + 1;
  mLowerBound = arma::vec(mInputDimension);
  mUpperBound = arma::vec(mInputDimension);
  double limitValue = std::numeric_limits<double>::max();

  set_lower_bound(
    arma::join_cols(arma::join_cols(mFKernel.get_lower_bound(),
                                    mSKernel.get_lower_bound()),
                                    arma::join_cols(rep_value(mHeadX.min(), mNoPseudoInputs),
                                                    arma::vec({-limitValue}))
    ));
  set_upper_bound(
    arma::join_cols(arma::join_cols(mFKernel.get_upper_bound(),
                                    mSKernel.get_upper_bound()),
                                    arma::join_cols(rep_value(mHeadX.max(), mNoPseudoInputs),
                                                    arma::vec({limitValue}))
    ));

  calculate_model_matrices();

  // set initial values for the posterior parameters
  mPosteriorFMean.resize(mNoPseudoInputs);
  mPosteriorFMean.fill(0);
  mPosteriorFCov = mKmm;
  mPosteriorSCov = mJmm;
  mPosteriorSMean.resize(mNoPseudoInputs);
  mPosteriorSMean.fill(mV);

  mLikelihoodLowerBounds.reserve(2 * mMaxIt);
  mLikelihoodLowerBounds.push_back(get_lower_bound());
  if (mVerbose) {
    std::cout << "Initial Lower Bound L = " << mLikelihoodLowerBounds[0] << std::endl;
  }

  int nIt = 1;
  double previousL = mLikelihoodLowerBounds[0];
  for (; nIt <= mMaxIt; nIt++) {
    update_distributions();
    mLikelihoodLowerBounds.push_back(get_lower_bound());
    if (mVerbose) {
      distributions_update_message(nIt);
    }
    // optimize hyperparams
    arma::vec hp = zip_hyperparameters();
    mLowerBoundSolver.optimize(*this, hp);
    unzip_hyperparameters(hp);
    calculate_model_matrices();
    mLikelihoodLowerBounds.push_back( get_lower_bound() );
    if (mVerbose) {
      hyperparameters_optimization_message(nIt);
      print_vector(hp, "HP = ");
      Rcpp::Rcout << std::endl;
    }
    if (std::fabs((mLikelihoodLowerBounds.back() - previousL) / previousL) < mRelativeTolerance) {
      if (mVerbose) {
        Rcpp::Rcout << "CONVERGENCE" << std::endl;
      }
      break;
    }
    previousL = mLikelihoodLowerBounds.back();
  }

  return Rcpp::List::create(Rcpp::Named("Ls") = mLikelihoodLowerBounds,
                            Rcpp::Named("fMean") = mPosteriorFMean,
                            Rcpp::Named("sMean") = mPosteriorSMean,
                            Rcpp::Named("fCov") = mPosteriorFCov,
                            Rcpp::Named("sCov") = mPosteriorSCov,
                            Rcpp::Named("xm") = mXm,
                            Rcpp::Named("fHp") = mFKernel.get_hyperparams(),
                            Rcpp::Named("sHp") = mSKernel.get_hyperparams(),
                            Rcpp::Named("v") = mV
  );
}

arma::colvec sde_variational_inferencer::calculate_E_vector() const {
  return calculate_E_vector(mV, mPosteriorSMean, mPosteriorSCov, mB, mHii);
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v,
                                                            const arma::vec& sMean,
                                                            const arma::mat& sCov,
                                                            const arma::mat& B,
                                                            const arma::vec& Hii) {
  int noPseudoInputs = sMean.n_elem;
  int dxSize = Hii.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(sMean.begin(), sMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  arma::colvec BgMean = B * sMeanMinusV;
  arma::mat gCovByB = sCov * B.t();
  arma::colvec Evector(dxSize);
  for (int i = 0; i < dxSize; i++) {
    double bb = 0;
    for (int j = 0; j < noPseudoInputs; j++) {
      bb += B(i,j) * gCovByB(j,i);
    }
    Evector(i) = std::exp(-v - BgMean(i) + 0.5 * (bb + Hii(i)));
  }
  return Evector;
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector() const {
  return calculate_ksi_vector(mTarget, mSamplingPeriod, mPosteriorFMean,
                              mPosteriorFCov, mA, mQii);
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector(
    const arma::vec& dxTarget, double samplingPeriod, const arma::vec& fMean,
    const arma::mat& fCov, const arma::mat& A, const arma::vec& Qii) {
  int dxSize = Qii.n_elem;
  arma::colvec Af = A * fMean;
  arma::mat Fmat = fCov + fMean * fMean.t();
  double h2 = samplingPeriod * samplingPeriod;

  arma::colvec result(dxSize);
  for (int i = 0; i < dxSize; i++) {
    arma::rowvec ai = A.row(i);
    result(i) = std::pow(dxTarget(i), 2) - 2 * samplingPeriod * dxTarget(i) * Af(i) +
      h2 * (arma::dot(ai, Fmat * ai.t()) +  Qii(i));
  }
  return result;
}

double sde_variational_inferencer::get_lower_bound() const {
  return get_lower_bound(mTarget, mSamplingPeriod,
                         mPosteriorFMean, mPosteriorFCov, mA, mQii, mKmmInv, mV,
                         mPosteriorSMean, mPosteriorSCov, mB, mHii, mJmmInv);
}

double sde_variational_inferencer::get_lower_bound(const arma::vec& dxTarget,
                                                   double samplingPeriod,
                                                   const arma::vec& fMean,
                                                   const arma::mat& fCov,
                                                   const arma::mat& A,
                                                   const arma::vec& Qii,
                                                   const arma::mat& kmmInv,
                                                   double v,
                                                   const arma::vec& sMean,
                                                   const arma::mat& sCov,
                                                   const arma::mat& B,
                                                   const arma::vec& Hii,
                                                   const arma::mat& jmmInv) {
  arma::vec E = calculate_E_vector(v, sMean, sCov, B, Hii);
  arma::vec ksi = calculate_ksi_vector(dxTarget, samplingPeriod, fMean, fCov, A, Qii);

  int noPseudoInputs = fMean.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(sMean.begin(), sMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  int dxSize = dxTarget.n_elem;
  double log2pi = log(2 * arma::datum::pi );
  double logEntConstant = noPseudoInputs * log(2 * arma::datum::pi * arma::datum::e);

  return  0.5 * (-arma::dot(E, ksi) / samplingPeriod
                 -dxSize * v - sum(B * sMeanMinusV) +
                 -dxSize * log(samplingPeriod) - dxSize * log2pi +
                 -arma::trace(jmmInv * sCov) +
                 -arma::dot(sMeanMinusV, jmmInv * sMeanMinusV) +
                 -noPseudoInputs * log2pi  + log(arma::det(jmmInv))
                 -arma::trace(kmmInv * fCov) +
                 -arma::dot(fMean, kmmInv * fMean) +
                 -noPseudoInputs * log2pi  + log(arma::det(kmmInv)) +
                 log(det(fCov)) + log(det(sCov)) + 2 * logEntConstant);
}

double sde_variational_inferencer::operator()(const arma::vec& x) {
  mFKernel.set_hyperparams(x.subvec(0, mNoFHyperparameters - 1));
  mSKernel.set_hyperparams(x.subvec(mNoFHyperparameters,
                                    mNoFHyperparameters + mNoSHyperparameters - 1));
  arma::vec xmVector = x.subvec(mNoFHyperparameters + mNoSHyperparameters, x.n_elem - 2);
  arma::mat xm = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
  double v = x(x.n_elem - 1);

  try {
    calculate_kernel_matrices(mHeadX, mFKernel, xm,
                              mKmm, mKmmInv, mKnm, mA, mQii);
    calculate_kernel_matrices(mHeadX, mSKernel,
                              xm, mJmm, mJmmInv, mJnm, mB, mHii);
    } catch (const std::exception& e) {
    // there is some singular matrix due to a bad setting of hyperparams.
    // return worst value, which is a large number since the solver minimizes.
    // However, there is some funny error with the l-bfgs-b algorithm if we
    // return a very big value (e.g., numeric_limits<double>::max... So it is
    // better to return a smallest one.
    // See http://stackoverflow.com/questions/5708480/problem-with-64-bit-rs-optim-under-windows-7
    return std::numeric_limits<float>::max();
  }
  // note the minus sign due to the fact that the solver minimizes and we
  // want to maximize
  return -get_lower_bound(mTarget, mSamplingPeriod, mPosteriorFMean,
                          mPosteriorFCov, mA, mQii, mKmmInv, v,
                          mPosteriorSMean, mPosteriorSCov,
                          mB, mHii, mJmmInv);

}

void sde_variational_inferencer::calculate_model_matrices() {
  calculate_kernel_matrices(mHeadX, mFKernel, mXm, mKmm, mKmmInv, mKnm, mA, mQii);
  calculate_kernel_matrices(mHeadX, mSKernel, mXm, mJmm, mJmmInv, mJnm, mB, mHii);
}


void sde_variational_inferencer::calculate_kernel_matrices(const arma::mat& headX,
                                                           const kernel& ker,
                                                           const arma::mat& xm,
                                                           arma::mat& kmm,
                                                           arma::mat& kmmInv,
                                                           arma::mat& knm,
                                                           arma::mat& A,
                                                           arma::vec& Qii) {
  kmm = ker.autocovmat(xm);
  kmmInv = arma::inv_sympd(kmm);
  // stabilize computations
  kmmInv = (kmmInv + kmmInv.t()) / 2.0;
  knm = ker.covmat(headX, xm);
  A = knm * kmmInv;

  int dxSize = headX.n_rows;
  arma::colvec auxVec = arma::zeros<arma::colvec>(dxSize);
  for (int i = 0; i < dxSize; i++) {
    for (int j = 0; j < xm.n_rows; j++) {
      for (int k = 0; k < xm.n_rows; k++) {
        auxVec(i) += knm(i,j) * kmmInv(j,k) * knm(i,k);
      }
    }
  }
  Qii = ker.variances(headX) - auxVec;
  Qii(find(Qii < 0)).zeros();
}

void sde_variational_inferencer::update_distributions() {
  // Update distribution of the drift
  arma::vec E = calculate_E_vector();
  arma::mat posteriorFCovInv = mKmmInv + mSamplingPeriod * mA.t() * diagmat(E) * mA;
  posteriorFCovInv = ( posteriorFCovInv + posteriorFCovInv.t() ) / 2.0;
  mPosteriorFCov = arma::inv_sympd(posteriorFCovInv);
  mPosteriorFCov = (mPosteriorFCov + mPosteriorFCov.t()) / 2.0;
  mPosteriorFMean = ( ((E % mTarget).t() * mA) * mPosteriorFCov).t();

  // Update distribution of the diffusion
  arma::vec ksi = calculate_ksi_vector();
  arma::vec logksi = arma::log(ksi);
  std::transform(logksi.begin(), logksi.end(),
                 logksi.begin(),
                 bind2nd(std::plus<double>(), -mV));

  // use laplace approximation to find the distribution of the diffusion
  arma::vec cvector = arma::exp(logksi + mHii / 2.0);

  laplace_objective_function lof(cvector, *this);
  mLaplaceSolver.optimize(lof, mPosteriorSMean);

  arma::mat Bp = mB, Bpp = mB;
  arma::vec meanMinusV(mPosteriorSMean.n_elem);
  std::transform(mPosteriorSMean.begin(), mPosteriorSMean.end(),
                 meanMinusV.begin(),
                 bind2nd(std::plus<double>(), -mV));
  arma::vec auxVec = arma::exp(-mB * meanMinusV);
  for (int i = 0; i < mB.n_cols; i++) {
    Bp.col(i) = mB.col(i) % cvector;
    Bpp.col(i) = mB.col(i) % auxVec;
  }
  arma::mat auxMat = 1.0 / (2.0 * mSamplingPeriod) * (Bp.t() * Bpp) + mJmmInv;
  auxMat = (auxMat + auxMat.t()) / 2.0;
  mPosteriorSCov = arma::inv_sympd(auxMat);
  mPosteriorSCov = (mPosteriorSCov + mPosteriorSCov.t()) / 2.0;
}

arma::vec sde_variational_inferencer::zip_hyperparameters() const {
  return arma::join_cols(
    arma::join_cols(mFKernel.get_hyperparams(), mSKernel.get_hyperparams()),
    arma::join_cols(matrix_to_vector(mXm), arma::vec({mV}))
  );
}

void sde_variational_inferencer::unzip_hyperparameters(const arma::vec& hp) {
  mFKernel.set_hyperparams(hp.subvec(0, mNoFHyperparameters - 1));
  mSKernel.set_hyperparams(hp.subvec(mNoFHyperparameters,
                                     mNoFHyperparameters + mNoSHyperparameters - 1));
  arma::vec xmVector = hp.subvec(mNoFHyperparameters + mNoSHyperparameters,
                                 hp.n_elem - 2);
  mXm = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
  mV = hp(hp.n_elem - 1);
}


void sde_variational_inferencer::distributions_update_message(int nIt) {
  Rcpp::Rcout << std::setprecision(8) <<
    "Iteration " << nIt << "| Distributions update | L = " <<
      mLikelihoodLowerBounds[2 * (nIt - 1) + 1] << std::endl;
}

void sde_variational_inferencer::hyperparameters_optimization_message(int nIt) {
  Rcpp::Rcout << std::setprecision(8) <<
    "Iteration " << nIt << "| Hyperparameter optimization | L = " <<
      mLikelihoodLowerBounds[2 * nIt] << std::endl;
}

void sde_variational_inferencer::print_vector(const arma::vec& x,
                                              const std::string& preface) {
  Rcpp::Rcout << std::setprecision(3) << preface;
  for (int i = 0; i < x.n_elem; i++) {
    Rcpp::Rcout << x[i] << " ";
  }
  Rcpp::Rcout << std::endl;
}


sde_variational_inferencer::laplace_objective_function::laplace_objective_function(
  const arma::vec& cvector,
  const sde_variational_inferencer& inferencer)
  : problem(inferencer.mNoPseudoInputs),
    mCVector(cvector),
    mInferencer(inferencer) {
}

double sde_variational_inferencer::laplace_objective_function::operator() (
    const arma::vec& x) {
  arma::colvec meanMinusV(x.n_elem);
  std::transform(x.begin(), x.end(), meanMinusV.begin(),
                 bind2nd(std::plus<double>(), -mInferencer.mV));
  arma::vec tmp = mInferencer.mB * meanMinusV;
  return -0.5 * (-1.0 / mInferencer.mSamplingPeriod *
                 arma::dot(mCVector, arma::exp(-tmp)) -
                 arma::accu(tmp) - arma::dot(meanMinusV, mInferencer.mJmmInv * meanMinusV));
}
