#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <vector>
#include <algorithm>


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

/////////////////////////////////

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


void generate_matrix(size_t n, uint_fast64_t seed, CSR_matrix &ans) {
    std::mt19937 rg(seed);
    std::uniform_int_distribution<> dist(-n/2, n/2);

    ans.set_size(n);
    tridiagonal_matrix a(n);
    for (size_t i = 0; i < n-1; ++i) {
        a.middle[i] = dist(rg);
        a.upper[i] = dist(rg);
        a.lower[i] = dist(rg);
    }
    *a.middle.rbegin() = dist(rg);

    from_tridiagonal(a, ans);
}


void generate_vector(
        size_t n, uint_fast64_t seed,
        std::vector<double> &ans) {

    std::mt19937 rg(seed);
    std::uniform_real_distribution<> dist(-1, 1);

    ans.resize(n, 0);
    for (size_t i = 0; i < n; ++i) {
        ans[i] = dist(rg);
    }
}

/////////////////////////////////

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

/////////////////////////////////

std::vector<double> control_sum(
        const std::vector<double> &x,
        uint_fast64_t seed) {
    // sum, L2, min, max, dot(x, random vector)
    std::vector<double> ans;
    ans.push_back(0);
    ans.push_back(0);
    ans.push_back(x.back());
    ans.push_back(x.back());
    for (auto i : x) {
        ans[0] += i;
        ans[1] += i*i;
        ans[2] = std::min(ans[2], i);
        ans[3] = std::max(ans[3], i);
    }
    ans[1] = std::sqrt(ans[1]);

    std::vector<double> h;
    generate_vector(x.size(), seed, h);
    ans.push_back(dot_product(h, x));
    return ans;
}


#define PRINT_LINEAR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

int main(int argc, char **args) {
    int N = 0;
    uint_fast64_t seed = 22, cs_seed = 999;
    std::vector<double> v, pr, h;

    std::cout.precision(16);

    if (argc < 2) {
        N = 10;
    }
    else {
        N = std::stoi(args[1]);
        if (argc >= 3) {
            seed = std::stoi(args[2]);
            if (argc >= 4) {
                cs_seed = std::stoi(args[3]);
            }
        }
    }

    bool debug = (N <= 10);
    CSR_matrix b(N);


    std::cout << "Usage: " << args[0] << " N [seed] [control_sum_seed]\n\n";
    std::cout << "N : " << N << std::endl;

    generate_matrix(N, 10 + seed, b);
    if (debug) {
        PRINT_LINEAR(b.IA, "IA");
        PRINT_LINEAR(b.JA, "JA");
        PRINT_LINEAR(b.values, "A ");

        std::cout << "[\n";
        for (int i = 0; i < N; ++i) {
            std::cout << " [";
            for (int j = 0; j < N; ++j) {
                std::cout << b.getv(i, j) << ",  ";
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";
    }

    generate_vector(N, 1+seed, v);
    product(b, v, pr);
    if (debug) {
        PRINT_LINEAR(v, "v  ");
        PRINT_LINEAR(pr, "A v");
    }
    PRINT_LINEAR(control_sum(pr, cs_seed), "control sum of A v");

    generate_vector(N, 2+seed, h);
    if (debug) {
        PRINT_LINEAR(h, "h  ");
        std::cout << "(h, v) = " << dot_product(h, v) << "\n";
    }

    linear_combination(3.14, 2.78, h, v);
    if (debug) {
        PRINT_LINEAR(h, "h = 3.14*h + 2.78*v");
    }
    PRINT_LINEAR(control_sum(h, cs_seed), "control sum of h");
}
