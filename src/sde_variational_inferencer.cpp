#define ARMA_DONT_PRINT_ERRORS
#include "sde_variational_inferencer.h"
#include "utils.h"
#include <cassert>

sde_variational_inferencer::sde_variational_inferencer(kernel& driftKernel,
                                                       kernel& diffKernel,
                                                       double v)
  :  problem(),
     mDriftKernel((assert(driftKernel.get_input_dimension() == diffKernel.get_input_dimension()),
               driftKernel)),
               mDiffKernel(diffKernel), mV(v),
               mNoDriftHyperparameters(driftKernel.get_hyperparams().n_elem),
               mNoDiffHyperparameters(diffKernel.get_hyperparams().n_elem),
               mLaplaceSolver(),
               mLowerBoundSolver() {
  mLaplaceSolver.set_projected_gradient_tolerance(0);
  mLowerBoundSolver.set_projected_gradient_tolerance(0);
  mLowerBoundSolver.set_max_iterations(5);
}

sde_variational_inferencer::sde_variational_inferencer(kernel& driftKernel,
                                                       kernel& diffKernel,
                                                       double v,
                                                       l_bfgs_b<arma::vec>& laplaceSolver,
                                                       l_bfgs_b<arma::vec>& lowerBoundSolver)
  : problem(),
    mDriftKernel((assert(driftKernel.get_input_dimension() == diffKernel.get_input_dimension()),
              driftKernel)),
              mDiffKernel(diffKernel), mV(v),
              mNoDriftHyperparameters(driftKernel.get_hyperparams().n_elem),
              mNoDiffHyperparameters(diffKernel.get_hyperparams().n_elem),
              mLaplaceSolver(laplaceSolver),
              mLowerBoundSolver(lowerBoundSolver) {
}

int sde_variational_inferencer::get_hyperparams_iterations() const {
  return mLowerBoundSolver.get_max_iterations();
}

void sde_variational_inferencer::set_hyperparams_iterations(int hyperparamsIterations) {
  if (hyperparamsIterations <= 0) {
    throw std::invalid_argument("The maximum number of iterations should be > 0");
  }
  mLowerBoundSolver.set_max_iterations(hyperparamsIterations);
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
                                                    arma::mat& inducingPoints,
                                                    int targetIndex){
  if (timeSeries.n_cols != mDriftKernel.get_input_dimension()) {
    throw std::invalid_argument("The time series' dimension does not match \
                                 the kernels' input dimension");
  }
  if (timeSeries.n_cols != inducingPoints.n_cols) {
    throw std::invalid_argument("The time series' dimension does not match \
                                 the inducing points' dimension");
  }
  if (inducingPoints.n_rows < 2) {
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
  mNoPseudoInputs = inducingPoints.n_rows;
  mInducingPoints = inducingPoints;

  Rcpp::Rcout << "Let's do some VI" << std::endl;
  mDriftKernel.greet();

  // set the variables related to the class problem properly
  // TODO: create a protected set_input_dimension that resets also
  // the bounds
  mInputDimension = mNoDriftHyperparameters + mNoDiffHyperparameters + mNoPseudoInputs + 1;
  mLowerBound = arma::vec(mInputDimension);
  mUpperBound = arma::vec(mInputDimension);
  double limitValue = std::numeric_limits<double>::max();
  // avoid consistency problems when trying to set the new parameters
  mLowerBound.fill(-limitValue);
  mUpperBound.fill(limitValue);

  set_lower_bound(
    arma::join_cols(arma::join_cols(mDriftKernel.get_lower_bound(),
                                    mDiffKernel.get_lower_bound()),
                                    arma::join_cols(rep_value(mHeadX.min(), mNoPseudoInputs),
                                                    arma::vec({-limitValue}))
    ));
  set_upper_bound(
    arma::join_cols(arma::join_cols(mDriftKernel.get_upper_bound(),
                                    mDiffKernel.get_upper_bound()),
                                    arma::join_cols(rep_value(mHeadX.max(), mNoPseudoInputs),
                                                    arma::vec({limitValue}))
    ));

  calculate_model_matrices();

  // set initial values for the posterior parameters
  mInducingDriftMean.resize(mNoPseudoInputs);
  mInducingDriftMean.fill(0);
  mInducingDriftCov = mKmm;
  mInducingDiffCov = mJmm;
  mInducingDiffMean.resize(mNoPseudoInputs);
  mInducingDiffMean.fill(mV);

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
    //print_vector(hp, "HP: ");
    //print_vector(mLowerBound, "LB: ");
    //print_vector(mUpperBound, "UB: ");

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
                            Rcpp::Named("fMean") = mInducingDriftMean,
                            Rcpp::Named("sMean") = mInducingDiffMean,
                            Rcpp::Named("fCov") = mInducingDriftCov,
                            Rcpp::Named("sCov") = mInducingDiffCov,
                            Rcpp::Named("xm") = mInducingPoints,
                            Rcpp::Named("fHp") = mDriftKernel.get_hyperparams(),
                            Rcpp::Named("sHp") = mDiffKernel.get_hyperparams(),
                            Rcpp::Named("v") = mV
  );
}

double sde_variational_inferencer::operator()(const arma::vec& x) {
  mDriftKernel.set_hyperparams(x.subvec(0, mNoDriftHyperparameters - 1));
  mDiffKernel.set_hyperparams(x.subvec(mNoDriftHyperparameters,
                                    mNoDriftHyperparameters + mNoDiffHyperparameters - 1));
  arma::vec xmVector = x.subvec(mNoDriftHyperparameters + mNoDiffHyperparameters, x.n_elem - 2);
  arma::mat xm = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
  double v = x(x.n_elem - 1);

  try {
    calculate_kernel_matrices(mHeadX, mDriftKernel, xm,
                              mKmm, mKmmInv, mKnm, mA, mQii);
    calculate_kernel_matrices(mHeadX, mDiffKernel,
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
  return -get_lower_bound(mTarget, mSamplingPeriod, mInducingDriftMean,
                          mInducingDriftCov, mA, mQii, mKmmInv, v,
                          mInducingDiffMean, mInducingDiffCov,
                          mB, mHii, mJmmInv);

}

double sde_variational_inferencer::get_lower_bound() const {
  return get_lower_bound(mTarget, mSamplingPeriod,
                         mInducingDriftMean, mInducingDriftCov, mA, mQii, mKmmInv, mV,
                         mInducingDiffMean, mInducingDiffCov, mB, mHii, mJmmInv);
}

double sde_variational_inferencer::get_lower_bound(const arma::vec& dxTarget,
                                                   double samplingPeriod,
                                                   const arma::vec& driftMean,
                                                   const arma::mat& driftCov,
                                                   const arma::mat& A,
                                                   const arma::vec& Qii,
                                                   const arma::mat& kmmInv,
                                                   double v,
                                                   const arma::vec& diffMean,
                                                   const arma::mat& diffCov,
                                                   const arma::mat& B,
                                                   const arma::vec& Hii,
                                                   const arma::mat& jmmInv) const {
  arma::vec E = calculate_E_vector(v, diffMean, diffCov, B, Hii);
  arma::vec ksi = calculate_ksi_vector(dxTarget, samplingPeriod, driftMean, driftCov, A, Qii);

  int noPseudoInputs = driftMean.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(diffMean.begin(), diffMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  int dxSize = dxTarget.n_elem;
  double log2pi = log(2 * arma::datum::pi );
  double logEntConstant = noPseudoInputs * log(2 * arma::datum::pi * arma::datum::e);

  return  0.5 * (-arma::dot(E, ksi) / samplingPeriod
                 -dxSize * v - sum(B * sMeanMinusV) +
                 -dxSize * log(samplingPeriod) - dxSize * log2pi +
                 -arma::trace(jmmInv * diffCov) +
                 -arma::dot(sMeanMinusV, jmmInv * sMeanMinusV) +
                 -noPseudoInputs * log2pi  + log(arma::det(jmmInv))
                 -arma::trace(kmmInv * driftCov) +
                 -arma::dot(driftMean, kmmInv * driftMean) +
                 -noPseudoInputs * log2pi  + log(arma::det(kmmInv)) +
                 log(det(driftCov)) + log(det(diffCov)) + 2 * logEntConstant);
}


void sde_variational_inferencer::calculate_model_matrices() {
  calculate_kernel_matrices(mHeadX, mDriftKernel, mInducingPoints, mKmm, mKmmInv, mKnm, mA, mQii);
  calculate_kernel_matrices(mHeadX, mDiffKernel, mInducingPoints, mJmm, mJmmInv, mJnm, mB, mHii);
}

void sde_variational_inferencer::update_distributions() {
  // Update distribution of the drift
  arma::vec E = calculate_E_vector();
  arma::mat posteriorFCovInv = mKmmInv + mSamplingPeriod * mA.t() * diagmat(E) * mA;
  posteriorFCovInv = ( posteriorFCovInv + posteriorFCovInv.t() ) / 2.0;
  mInducingDriftCov = arma::inv_sympd(posteriorFCovInv);
  mInducingDriftCov = (mInducingDriftCov + mInducingDriftCov.t()) / 2.0;
  mInducingDriftMean = ( ((E % mTarget).t() * mA) * mInducingDriftCov).t();

  // Update distribution of the diffusion
  arma::vec ksi = calculate_ksi_vector();
  arma::vec logksi = arma::log(ksi);
  std::transform(logksi.begin(), logksi.end(),
                 logksi.begin(),
                 bind2nd(std::plus<double>(), -mV));

  // use laplace approximation to find the distribution of the diffusion
  arma::vec cvector = arma::exp(logksi + mHii / 2.0);

  laplace_objective_function lof(cvector, *this);
  mLaplaceSolver.optimize(lof, mInducingDiffMean);

  arma::mat Bp = mB, Bpp = mB;
  arma::vec meanMinusV(mInducingDiffMean.n_elem);
  std::transform(mInducingDiffMean.begin(), mInducingDiffMean.end(),
                 meanMinusV.begin(),
                 bind2nd(std::plus<double>(), -mV));
  arma::vec auxVec = arma::exp(-mB * meanMinusV);
  for (int i = 0; i < mB.n_cols; i++) {
    Bp.col(i) = mB.col(i) % cvector;
    Bpp.col(i) = mB.col(i) % auxVec;
  }
  arma::mat auxMat = 1.0 / (2.0 * mSamplingPeriod) * (Bp.t() * Bpp) + mJmmInv;
  auxMat = (auxMat + auxMat.t()) / 2.0;
  mInducingDiffCov = arma::inv_sympd(auxMat);
  mInducingDiffCov = (mInducingDiffCov + mInducingDiffCov.t()) / 2.0;
}

void sde_variational_inferencer::calculate_kernel_matrices(const arma::mat& headX,
                                                           const kernel& ker,
                                                           const arma::mat& inducingPoints,
                                                           arma::mat& kmm,
                                                           arma::mat& kmmInv,
                                                           arma::mat& knm,
                                                           arma::mat& A,
                                                           arma::vec& Qii) {
  kmm = ker.autocovmat(inducingPoints);
  kmmInv = arma::inv_sympd(kmm);
  // stabilize computations
  kmmInv = (kmmInv + kmmInv.t()) / 2.0;
  knm = ker.covmat(headX, inducingPoints);
  A = knm * kmmInv;

  int dxSize = headX.n_rows;
  arma::colvec auxVec = arma::zeros<arma::colvec>(dxSize);
  for (int i = 0; i < dxSize; i++) {
    for (int j = 0; j < inducingPoints.n_rows; j++) {
      for (int k = 0; k < inducingPoints.n_rows; k++) {
        auxVec(i) += knm(i,j) * kmmInv(j,k) * knm(i,k);
      }
    }
  }
  Qii = ker.variances(headX) - auxVec;
  Qii(find(Qii < 0)).zeros();
}

arma::colvec sde_variational_inferencer::calculate_E_vector() const {
  return calculate_E_vector(mV, mInducingDiffMean, mInducingDiffCov, mB, mHii);
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v,
                                                            const arma::vec& diffMean,
                                                            const arma::mat& diffCov,
                                                            const arma::mat& B,
                                                            const arma::vec& Hii) const {
  int noPseudoInputs = diffMean.n_elem;
  int dxSize = Hii.n_elem;
  arma::colvec sMeanMinusV(noPseudoInputs);
  std::transform(diffMean.begin(), diffMean.end(),
                 sMeanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));

  arma::colvec BgMean = B * sMeanMinusV;
  arma::mat gCovByB = diffCov * B.t();
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
  return calculate_ksi_vector(mTarget, mSamplingPeriod, mInducingDriftMean,
                              mInducingDriftCov, mA, mQii);
}

arma::colvec sde_variational_inferencer::calculate_ksi_vector(
    const arma::vec& dxTarget, double samplingPeriod, const arma::vec& fMean,
    const arma::mat& fCov, const arma::mat& A, const arma::vec& Qii) const{
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

arma::vec sde_variational_inferencer::zip_hyperparameters() const {
  return arma::join_cols(
          arma::join_cols(mDriftKernel.get_hyperparams(), mDiffKernel.get_hyperparams()),
          arma::join_cols(matrix_to_vector(mInducingPoints), arma::vec({mV}))
  );
}

void sde_variational_inferencer::unzip_hyperparameters(const arma::vec& hyperparams) {
  mDriftKernel.set_hyperparams(hyperparams.subvec(0, mNoDriftHyperparameters - 1));
  mDiffKernel.set_hyperparams(hyperparams.subvec(mNoDriftHyperparameters,
                                     mNoDriftHyperparameters + mNoDiffHyperparameters - 1));
  arma::vec xmVector = hyperparams.subvec(mNoDriftHyperparameters + mNoDiffHyperparameters,
                                 hyperparams.n_elem - 2);
  mInducingPoints = vector_to_matrix(xmVector, mNoPseudoInputs, mHeadX.n_cols);
  mV = hyperparams(hyperparams.n_elem - 1);
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
