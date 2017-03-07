#include "sde_variational_inferencer.h"

sde_variational_inferencer::sde_variational_inferencer(const arma::mat& timeSeries,
                                                       double samplingPeriod)
  : mX(timeSeries),
    mHeadX(timeSeries.submat(0, 0, timeSeries.n_rows - 2, timeSeries.n_cols - 1)),
    mDX(arma::diff(timeSeries, 1)),
    mSamplingPeriod(samplingPeriod),
    mDXSize(timeSeries.n_rows - 1) {

}


void sde_variational_inferencer::do_inference(int targetIndex, arma::mat& xm,
                                              kernel& fKernel, kernel& sKernel,
                                              double v){
  Rcpp::Rcout << "Let's do some VI" << std::endl;
  // TODO: sanitize inputs
  mTarget = mDX.col(targetIndex);
  mNoPseudoInputs = xm.n_rows;
  mNoFHyperparameters = fKernel.get_hyperparams().n_elem;
  mNoSHyperparameters = sKernel.get_hyperparams().n_elem;

  calculate_model_matrices(fKernel, sKernel, xm);

  mPosteriorFMean.resize(mNoPseudoInputs);
  mPosteriorFMean.fill(0);
  mPosteriorFCov = mKmm;
  mPosteriorSCov = mJmm;
  mPosteriorSMean.resize(mNoPseudoInputs);
  mPosteriorSMean.fill(v);
  mLowerBounds.reserve(2 * mMaxIt);
  mLowerBounds.push_back(get_lower_bound(v));
  if (mVerbose) {
     std::cout << "Initial Lower Bound L = " << mLowerBounds[0] << std::endl;
  }
  int nIt = 1;
  double L, previousL = mLowerBounds[0];
  for (; nIt <= mMaxIt; nIt++) {
    update_distributions(fKernel, sKernel, v);
    mLowerBounds.push_back( get_lower_bound(v) );
    if (mVerbose) {
      distributions_update_message(nIt);
    }
    //optimize_hyperparameters();
    // mLowerBounds.push_back( get_lower_bound(v) );
    // if (mVerbose) {
    //  hyperparameters_optimization_message(nIt);
    //}
    if (std::fabs((mLowerBounds.back() - previousL) / previousL) < mRelativeTolerance) {
      if (mVerbose) {
        Rcpp::Rcout << "Convergence" << std::endl;
      }
      break;
    }
    previousL = mLowerBounds.back();
  }
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v) const {
  return calculate_E_vector(v, mPosteriorSMean, mPosteriorSCov, mB, mHii);
}

arma::colvec sde_variational_inferencer::calculate_E_vector(double v,
                                                            const arma::vec& sMean,
                                                            const arma::mat& sCov,
                                                            const arma::mat& B,
                                                            const arma::vec& Hii) {
 // Rcpp::Rcout << "E" << std::endl;
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

double sde_variational_inferencer::get_lower_bound(double v) const {
  return get_lower_bound(mTarget, mSamplingPeriod,
                         mPosteriorFMean, mPosteriorFCov, mA, mQii, mKmmInv, v,
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

void sde_variational_inferencer::calculate_model_matrices(const kernel& fKer,
                                                          const kernel& sKer,
                                                          const arma::mat& xm) {
  calculate_kernel_matrices(mHeadX, fKer, xm, mKmm, mKmmInv, mKnm, mA, mQii);
  calculate_kernel_matrices(mHeadX, sKer, xm, mJmm, mJmmInv, mJnm, mB, mHii);
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

void sde_variational_inferencer::update_distributions(const kernel& fKernel,
                                                      const kernel& sKernel,
                                                      double v) {
  // Update distribution of the drift
  arma::vec E = calculate_E_vector(v);
  arma::mat posteriorFCovInv = mKmmInv + mSamplingPeriod * mA.t() * diagmat(E) * mA;
  // Rcpp::Rcout << posteriorFCovInv << std::endl;
  posteriorFCovInv = ( posteriorFCovInv + posteriorFCovInv.t() ) / 2.0;
  mPosteriorFCov = arma::inv_sympd(posteriorFCovInv);
  mPosteriorFCov = (mPosteriorFCov + mPosteriorFCov.t()) / 2.0;
  mPosteriorFMean = ( ((E % mTarget).t() * mA) * mPosteriorFCov).t();

  // Update distribution of the diffusion
  arma::vec ksi = calculate_ksi_vector();
  arma::vec logksi = arma::log(ksi);
  std::transform(logksi.begin(), logksi.end(),
                 logksi.begin(),
                 bind2nd(std::plus<double>(), -v));

  // use laplace approximation to find the distribution of the diffusion
  arma::vec cvector = arma::exp(logksi + mHii / 2.0);

  laplace_objective_function lof(mPosteriorSMean.n_elem,
                                 cvector,  v,
                                 *this);
  mSolver.optimize(lof, mPosteriorSMean);

  arma::mat Bp = mB, Bpp = mB;
  arma::vec meanMinusV(mPosteriorSMean.n_elem);
  std::transform(mPosteriorSMean.begin(), mPosteriorSMean.end(),
                 meanMinusV.begin(),
                 bind2nd(std::plus<double>(), -v));
  arma::vec auxVec = arma::exp(-mB * meanMinusV);
  for (int i = 0; i < mB.n_cols; i++) {
    Bp.col(i) = mB.col(i) % cvector;
    Bpp.col(i) = mB.col(i) % auxVec;
  }
  arma::mat auxMat = 1.0 / (2.0 * mSamplingPeriod) * (Bp.t() * Bpp) + mJmmInv;
  auxMat = (auxMat + auxMat.t()) / 2.0;
  mPosteriorSCov = arma::inv_sympd(auxMat);
  mPosteriorSCov = (mPosteriorSCov + mPosteriorSCov.t()) / 2.0;

  // Rcpp::Rcout << "Fm = " << mPosteriorFMean << std::endl;
  // Rcpp::Rcout << "Sm= " << mPosteriorSMean << std::endl;
  // Rcpp::Rcout << "Fcov = " << mPosteriorFCov << std::endl;
  // Rcpp::Rcout << "Scov = " << mPosteriorSCov << std::endl;
  // Rcpp::Rcout << "********************" << std::endl << std::endl;

}


void sde_variational_inferencer::distributions_update_message(int nIt) {
  Rcpp::Rcout << "Iteration " << nIt << "| Distributions update | L = " <<
    mLowerBounds[nIt] << std::endl;
}

void sde_variational_inferencer::hyperparameters_optimization_message(int nIt) {
   Rcpp::Rcout << "Iteration " << nIt << "| Distributions update | L = " <<
    mLowerBounds[nIt + 1] << std::endl;
}