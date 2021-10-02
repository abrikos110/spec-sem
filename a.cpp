#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


struct tridiagonal_matrix {
    std::vector<double> upper, middle, lower;

    tridiagonal_matrix(size_t n)
        : upper(n-1, 0), middle(n, 0), lower(n-1, 0) {}

    size_t size() {
        return middle.size();
    }

    double &get(int i, int j) {
        if (i - j == 1) {
            return lower[j];
        }
        else if (i == j) {
            return middle[i];
        }
        else if (i - j == -1) {
            return upper[i];
        }
        throw "|i-j| > 1";
    }

    double getv(int i, int j) {
        return (-1 <= i-j && i-j <= 1) ? get(i, j) : 0;
    }
};


struct CSR_matrix {
    std::vector<size_t> IA, JA;
    std::vector<double> values;

    CSR_matrix(size_t n)
        : IA(n, 0) {}

    size_t size() {
        return IA.size();
    }

    size_t JA_begin(size_t i) {
        return IA[i];
    }
    size_t JA_end(size_t i) {
        return i < size() - 1 ? IA[i+1] : JA.size();
    }

    double getv(size_t i, size_t j) {
        for (size_t k = JA_begin(i); k < JA_end(i); ++k) {
            if (JA[k] == j) {
                return values[k];
            }
        }
        return 0;
    }
};


CSR_matrix from_tridiagonal(tridiagonal_matrix A) {
    CSR_matrix ans(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ans.IA[i] = ans.JA.size();
        for (size_t j = (i-1) * (i>0); j < A.size() && j < i+2; ++j) {
            if (A.getv(i, j) != 0) {
                ans.JA.push_back(j);
                ans.values.push_back(A.getv(i, j));
            }
        }
    }
    return ans;
}


CSR_matrix generate_matrix(size_t n) {
    tridiagonal_matrix a(n);
    for (size_t i = 0; i < n-1; ++i) {
        a.middle[i] = a.upper[i] = a.lower[i] = i;
    }
    *a.middle.rbegin() = n+5;

    return from_tridiagonal(a);
}

std::vector<double> generate_vector(size_t n) {
    std::vector<double> ans(n, 0);
    for (size_t i = 0; i < n; ++i) {
        ans[i] = (((~i) ^ 1379) % 97);
    }
    return ans;
}


std::vector<double> product(CSR_matrix A, std::vector<double> v) {
    std::vector<double> ans(A.size(), 0);
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t k = A.JA_begin(i); k < A.JA_end(i); ++k) {
            ans[i] += A.values[k] * v[A.JA[k]];
        }
    }
    return ans;
}


#define PRINT_LINEAR(x, name) do {std::cout << (name) << " : ";\
    for (auto iggg : (x)) std::cout << iggg << " ";\
    std::cout << std::endl; } while(0)

int main() {
    int N = 10;
    CSR_matrix b = generate_matrix(N);

    PRINT_LINEAR(b.IA, "IA");
    PRINT_LINEAR(b.JA, "JA");
    PRINT_LINEAR(b.values, "A ");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << b.getv(i, j) << "  ";
        }
        std::cout << "\n";
    }

    auto v = generate_vector(N);
    PRINT_LINEAR(v, "v  ");
    PRINT_LINEAR(product(b, v), "A v");
}
