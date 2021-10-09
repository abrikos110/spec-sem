#ifndef MATRICES_CPP
#define MATRICES_CPP

#include <vector>


struct tridiagonal_matrix {
    std::vector<double> upper, middle, lower;

    tridiagonal_matrix(size_t n)
        : upper(n-1, 0), middle(n, 0), lower(n-1, 0) {}

    size_t size() const {
        return middle.size();
    }

    double &get(size_t i, size_t j) {
        if (i - j == 1) {
            return lower[j];
        }
        else if (i == j) {
            return middle[i];
        }
        else if (j - i == 1) {
            return upper[i];
        }
        throw "|i-j| > 1";
    }

    double getv(size_t i, size_t j) const {
        if (i - j == 1) {
            return lower[j];
        }
        else if (i == j) {
            return middle[i];
        }
        else if (j - i == 1) {
            return upper[i];
        }
        return 0;
    }
};


struct CSR_matrix {
    std::vector<size_t> IA, JA;
    std::vector<double> values;

    CSR_matrix(size_t n = 0)
        : IA(n, 0) {}

    size_t size() const {
        return IA.size();
    }

    void set_size(size_t n) {
        IA.resize(n, 0);
    }

    size_t JA_begin(size_t i) const {
        return IA[i];
    }
    size_t JA_end(size_t i) const {
        return i < size() - 1 ? IA[i+1] : JA.size();
    }

    double getv(size_t i, size_t j) const {
        for (size_t k = JA_begin(i); k < JA_end(i); ++k) {
            if (JA[k] == j) {
                return values[k];
            }
        }
        return 0;
    }
};


void from_tridiagonal(const tridiagonal_matrix &A, CSR_matrix &ans) {
    for (size_t i = 0; i < A.size(); ++i) {
        ans.IA[i] = ans.JA.size();
        for (size_t j = (i-1) * (i>0); j < A.size() && j < i+2; ++j) {
            if (A.getv(i, j) != 0) {
                ans.JA.push_back(j);
                ans.values.push_back(A.getv(i, j));
            }
        }
    }
}


void product(
        const CSR_matrix &A,
        const std::vector<double> &v,
        std::vector<double> &ans) {

    ans.resize(A.size(), 0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t k = A.JA_begin(i); k < A.JA_end(i); ++k) {
            ans[i] += A.values[k] * v[A.JA[k]];
        }
    }
}


double dot_product(
        const std::vector<double> &a,
        const std::vector<double> &b) {

    double ans = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    return ans;
}


void linear_combination(double a, double b,
        std::vector<double> &x, const std::vector<double> &y) {
    // x = a * x + b * y
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = a * x[i] + b * y[i];
    }
}

#endif
