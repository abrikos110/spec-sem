#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


struct tridiagonal_matrix {
    std::vector<double> upper, middle, lower;

    tridiagonal_matrix(size_t n) {
        for (size_t i = 0; i < n-1; ++i) {
            upper.push_back(0);
            middle.push_back(0);
            lower.push_back(0);
        }
        middle.push_back(0);
    }

    size_t size() {
        return middle.size();
    }

    double &get(int i, int j) {
        if (i - j == 1) {
            return lower[i];
        }
        else if (i == j) {
            return middle[i];
        }
        else if (i - j == -1) {
            return upper[i];
        }
        throw "sth";
    }

    double getv(int i, int j) {
        if (-1 > i - j || i - j > 1) {
            return 0;
        }
        return get(i, j);
    }
};


struct CSR_matrix {
    std::vector<int> IA, JA;
    std::vector<double> values;
    size_t n;  // to be positive

    CSR_matrix(size_t n)
        : n(n) {
        IA.reserve(n+2);
        JA.reserve(3*n);
        values.reserve(3*n);
    }
};


CSR_matrix from_tridiagonal(tridiagonal_matrix A) {
    CSR_matrix ans(A.size());

    for (int row = 0; row < A.size(); ++row) {
        ans.IA.push_back(ans.JA.size());
        for (size_t j = std::max(0, row-1);
                j <= std::min((int)A.size() - 1, row + 1);
                ++j) {
            if (A.getv(row, j) != 0) {
                if (ans.IA.back() == -1) {
                    ++*ans.IA.rbegin();
                }
                ans.JA.push_back(j);
                ans.values.push_back(A.getv(row, j));
            }
        }
        if (ans.IA.back() == -1) {
            throw "zero row";
        }
    }

    return ans;
}




int main() {
    tridiagonal_matrix a(10);
    for (int i = 0; i < 10; ++i) {
        a.middle[i] = i+1;
        a.upper[i] = i+1;
        a.lower[i] = i+1;
    }

    CSR_matrix b = from_tridiagonal(a);

    std::cout << "IA : ";
    for (int i : b.IA) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "JA : ";
    for (int i : b.JA) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "A : ";
    for (int i : b.values) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}
