/*
 * Copyright Constantino Antonio Garcia 2017
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef L_BFGS_B_CPP_WRAPPER_H
#define L_BFGS_B_CPP_WRAPPER_H

#include <cassert>
#include "problem.h"
#include <vector>

extern "C" {
void setulb_wrapper(int *n, int *m, double x[], double l[], double u[], int nbd[], double *f,
                    double g[], double *factr, double *pgtol, double wa[], int iwa[], int *itask,
                    int *iprint, int *icsave, bool *lsave0, bool *lsave1, bool *lsave2, bool *lsave3,
                    int isave[], double dsave[]);
}

template<class T>
class l_bfgs_b {
public:
    l_bfgs_b() : l_bfgs_b(5) {

    };

    l_bfgs_b(int memorySize) : l_bfgs_b(memorySize, 500, 1e7, 1e-9) {

    }

    // Typical values for machinePrecisionFactor : 1e+12 for
    // low accuracy; 1e+7 for moderate accuracy; 1e+1 for extremely
    // high accuracy.
    l_bfgs_b(int memorySize, int maximumNumberOfIterations,
             double machinePrecisionFactor, double projectedGradientTolerance)
        // note the use of the , operator to check the correctness of the parameters
        : mMemorySize((check_memory_size(memorySize), memorySize)),
          mMaximumNumberOfIterations((check_max_iterations(maximumNumberOfIterations), maximumNumberOfIterations)),
          mMachinePrecisionFactor((check_precision_factor(machinePrecisionFactor), machinePrecisionFactor)),
          mProjectedGradientTolerance((check_gradient_tolerance(projectedGradientTolerance),
                  projectedGradientTolerance)),
          mVerboseLevel(-1) {
    }

    ~l_bfgs_b() = default;

    int get_memory_size() const {
        return mMemorySize;
    }

    void set_memory_size(int memorySize) {
        check_memory_size(memorySize);
        mMemorySize = memorySize;
    }

    int get_max_iterations() const {
        return mMaximumNumberOfIterations;
    }

    void set_max_iterations(int maximumNumberOfIterations) {
        check_max_iterations(maximumNumberOfIterations);
        mMaximumNumberOfIterations = maximumNumberOfIterations;
    }

    double get_machine_precision_factor() const {
        return mMachinePrecisionFactor;
    }

    void set_machine_precision_factor(double machinePrecisionFactor) {
        check_precision_factor(machinePrecisionFactor);
        mMachinePrecisionFactor = machinePrecisionFactor;
    }

    double get_projected_gradient_tolerance() const {
        return mProjectedGradientTolerance;
    }

    void set_projected_gradient_tolerance(double projectedGradientTolerance) {
        check_gradient_tolerance(projectedGradientTolerance);
        mProjectedGradientTolerance = projectedGradientTolerance;
    }

    int get_verbose_level() const {
        return mVerboseLevel;
    }

    void set_verbose_level(int verboseLevel) {
        // TODO: discretize verboseLevel as enum class
        mVerboseLevel = verboseLevel;
    }

    double get_gradient_scaling_factor() const {
       return mGradientScalingFactor;
    }

    void set_gradient_scaling_factor(double gradientScalingFactor) {
        if (gradientScalingFactor <= 0 || gradientScalingFactor > 1) {
            throw std::invalid_argument("gradientScalingFactor should be > 0 and <= 1");
        }
        mGradientScalingFactor = gradientScalingFactor;
    }

    void optimize(problem<T> &pb, T &x0) {
        int n = pb.get_input_dimension();
        // prepare variables for the algorithm
        std::vector<double> mLowerBound(n);
        std::vector<double> mUpperBound(n);
        std::vector<int> mNbd(n);
        std::vector<double> mWorkArray(2 * mMemorySize * n + 5 * n +
                                       11 * mMemorySize * mMemorySize + 8 * mMemorySize);
        std::vector<int> mIntWorkArray(3 * n);

        T lowerBound = pb.get_lower_bound();
        T upperBound = pb.get_upper_bound();
        bool hasLowerBound, hasUpperBound;
        for (int i = 0; i < n; ++i) {
            mLowerBound[i] = lowerBound[i];
            mUpperBound[i] = upperBound[i];
            hasLowerBound = !std::isinf(mLowerBound[i]);
            hasUpperBound = !std::isinf(mUpperBound[i]);
            // nbd(i)=0 if x(i) is unbounded,
            // 1 if x(i) has only a lower bound,
            // 2 if x(i) has both lower and upper bounds, and
            // 3 if x(i) has only an upper bound.
            if (hasLowerBound) {
                if (hasUpperBound) {
                    mNbd[i] = 2;
                } else {
                    mNbd[i] = 1;
                }
            } else if (hasUpperBound) {
                mNbd[i] = 3;
            } else {
                mNbd[i] = 0;
            }
        }

        double f = pb(x0);
        // use x0 to initialize gr with the proper dimensions without
        // dealing with Templates
        T gr(x0);
        pb.gradient(x0, gr);
        if (mGradientScalingFactor != 1.0) {
            scale_gradient(gr, n);
        }

        int i = 0;
        int itask = 0;
        int icsave = 0;

        bool test = false;
        // TODO: translate itask using enum class to make this more readable
        while ((i < mMaximumNumberOfIterations) && (
                (itask == 0) || (itask == 1) || (itask == 2) || (itask == 3)
        )) {
          setulb_wrapper(&n, &mMemorySize, &x0[0], &mLowerBound[0], &mUpperBound[0], &mNbd[0], &f,
                          &gr[0],
                         &mMachinePrecisionFactor, &mProjectedGradientTolerance,
                          &mWorkArray[0], &mIntWorkArray[0], &itask, &mVerboseLevel,
                          &icsave, &mBoolInformation[0], &mBoolInformation[1],
                          &mBoolInformation[2], &mBoolInformation[3],
                          &mIntInformation[0], &mDoubleInformation[0]);
            // assert that impossible values do not occur
            assert(icsave <= 14 && icsave >= 0);
            assert(itask <= 12 && itask >= 0);

            if (itask == 2 || itask == 3) {
                f = pb(x0);
                pb.gradient(x0, gr);
                if (mGradientScalingFactor != 1.0) {
                    scale_gradient(gr, n);
                }
            }

            i = mIntInformation[29];
        }

    }

private:
    int mMemorySize;
    double mMachinePrecisionFactor;
    double mProjectedGradientTolerance;
    int mVerboseLevel;
    int mMaximumNumberOfIterations;
    // factor <= 1 used to scale the gradient for explosive functions
    double mGradientScalingFactor = 1.0;
    // interface to Fortran code
    bool mBoolInformation[4];
    int mIntInformation[44];
    double mDoubleInformation[29];

    void scale_gradient(T& gradient, int gradientSize) {
        for (int i = 0; i < gradientSize; i++) {
            gradient[i] *= mGradientScalingFactor;
        }
    }

    static void check_memory_size(int memorySize) {
        if (memorySize < 1) {
            throw std::invalid_argument("memorySize should be >= 1");
        }
    }

    static void check_max_iterations(int maximumNumberOfIterations) {
        if (maximumNumberOfIterations < 1) {
            throw std::invalid_argument("maximumNumberOfIterations should be >= 1");
        }
    }

    static void check_precision_factor(double machinePrecisionFactor) {
        if (machinePrecisionFactor <= 0) {
            throw std::invalid_argument("machinePrecisionFactor should be > 0");
        }
    }

    static void check_gradient_tolerance(double projectedGradientTolerance) {
        if (projectedGradientTolerance < 0) {
            throw std::invalid_argument("projectedGradientTolerance should be >= 0");
        }
    }
};


#endif //L_BFGS_B_CPP_WRAPPER_H
