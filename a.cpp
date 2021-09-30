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
            return lower[j];
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

    CSR_matrix(size_t n) : n(n) {}

    double getv(int i, int j) {
        for (int k = IA[i]; k < (i+1 < IA.size() ? IA[i+1] : JA.size()); ++k) {
            if (JA[k] == j) {
                return values[k];
            }
        }
        return 0;
    }
};


CSR_matrix from_tridiagonal(tridiagonal_matrix A) {
    CSR_matrix ans(A.size());

    for (int i = 0; i < A.size(); ++i) {
        ans.IA.push_back(ans.JA.size());
        for (size_t j = std::max(0, i-1);
                j <= std::min((int)A.size() - 1, i + 1);
                ++j) {
            if (A.getv(i, j) != 0) {
                if (ans.IA.back() == -1) {
                    ++*ans.IA.rbegin();
                }
                ans.JA.push_back(j);
                ans.values.push_back(A.getv(i, j));
            }
        }
        if (ans.IA.back() == -1) {
            throw "zero row";
        }
    }

    return ans;
}


CSR_matrix generate_matrix(size_t n) {
    tridiagonal_matrix a(n);
    for (int i = 0; i < a.size()-1; ++i) {
        a.middle[i] = i+2;
        a.upper[i] = i+1;
        a.lower[i] = i+3;
    }
    *a.middle.rbegin() = n+5;

    return from_tridiagonal(a);
}



#define PRINT_LINEAR(x, name) do {std::cout << (name) << " : ";\
    for (int iggg : (x)) std::cout << iggg << " ";\
    std::cout << std::endl; } while(0)

int main() {
    CSR_matrix b = generate_matrix(10);

    PRINT_LINEAR(b.IA, "IA");
    PRINT_LINEAR(b.JA, "JA");
    PRINT_LINEAR(b.values, "A ");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << b.getv(i, j) << "  ";
        }
        std::cout << "\n";
    }
}
