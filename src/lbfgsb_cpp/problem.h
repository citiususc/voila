#ifndef LBFGSB_ADAPTER_PROBLEM_H
#define LBFGSB_ADAPTER_PROBLEM_H

#include <array>
#include <limits>

// Use the Curiosly repeating pattern to avoid code duplication
template<typename T, typename derived>
class problem_base {
public:
    problem_base(int inputDimension, const T &lowerBound, const T &upperBound) :
    // note the use of the , operator to check for the correctness of the arguments
            mInputDimension((check_input_dimension(inputDimension), inputDimension)),
            mLowerBound((check_container_dimensions(lowerBound.size(), inputDimension), lowerBound)),
            mUpperBound( (check_bounds(lowerBound, upperBound), upperBound)) {
    };

    problem_base(int inputDimension) :
            mInputDimension((check_input_dimension(inputDimension), inputDimension)),
            mLowerBound(inputDimension),
            mUpperBound(inputDimension) {
        set_default_bounds();
    }

    virtual ~problem_base() = default;

    int getInputDimension() const {
        return mInputDimension;
    }

    T get_lower_bound() const {
        return mLowerBound;
    }

    void set_lower_bound(const T &lowerBound) {
        check_container_dimensions(lowerBound.size(), mInputDimension);
        check_bounds(lowerBound, mUpperBound);
        mLowerBound = lowerBound;
    }

    T get_upper_bound() const {
        return mUpperBound;
    }

    void set_upper_bound(const T &upperBound) {
        check_container_dimensions(upperBound.size(), mInputDimension);
        check_bounds(mLowerBound, upperBound);
        mUpperBound = upperBound;
    }

    virtual double operator()(const T &x) = 0;

    virtual void gradient(const T &x, T &gr) = 0;

protected:
    int mInputDimension;
    T mLowerBound;
    T mUpperBound;

    problem_base() = default;

    void set_default_bounds() {
        for (int i = 0; i < mInputDimension; ++i) {
            mLowerBound[i] = -::std::numeric_limits<double>::infinity();
            mUpperBound[i] = ::std::numeric_limits<double>::infinity();
        }
    };

    static void check_input_dimension(int inputDimension) {
        if (inputDimension < 1) {
            throw std::invalid_argument("inputDimension should be >= 1");
        }
    }

    static void check_container_dimensions(int containerSize, int inputDimension) {
        if (containerSize != inputDimension) {
            throw std::invalid_argument("The container's size does not match the problem's input dimension");
        }
    }

    static void check_bounds(const T &lowerBound, const T &upperBound) {
        int n = lowerBound.size();
        if (n != upperBound.size()) {
            throw std::invalid_argument("lowerBound's size doest not match upperBound's size");
        }
        for (int i = 0; i < n; ++i) {
            if (lowerBound[i] > upperBound[i]) {
                throw std::invalid_argument("Incompatible bounds (lowerBound[i] > upperBound[i] for some i)");
            }
        }
    }
};

// The general version.
template<typename T>
class problem : public problem_base<T, problem<T> > {
private:
    typedef problem_base<T, problem<T> > base;

public:
    problem(int inputDimension, const T &lowerBound, const T &upperBound) :
            base(inputDimension, lowerBound, upperBound) {
    }

    problem(int inputDimension) : base(inputDimension) {
    }
};

// Specialization for std::array
template<typename U, std::size_t N>
class problem<std::array<U, N> > : public problem_base<std::array<U, N>, problem<std::array<U, N> > > {
private:
    typedef problem_base<std::array<U, N>, problem<std::array<U, N> > > base;

public:
    problem(int inputDimension, const std::array<U, N> &lowerBound, const std::array<U, N> &upperBound) :
            base() {
        this->check_input_dimension(inputDimension);
        this->mInputDimension = inputDimension;
        this->check_container_dimensions(N, inputDimension);
        this->check_bounds(lowerBound,upperBound);
        this->mLowerBound = lowerBound;
        this->mUpperBound = upperBound;
    }

    problem(int inputDimension) : base() {
        this->check_container_dimensions(N, inputDimension);
        this->mInputDimension = inputDimension;
        this->set_default_bounds();
    }
};

#endif //LBFGSB_ADAPTER_PROBLEM_H
