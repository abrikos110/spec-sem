#ifndef TESTS_CPP
#define TESTS_CPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "matrices.cpp"


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


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

void test_mat_vec_product(
        int N, uint_fast64_t seed,
        uint_fast64_t cs_seed, bool debug) {
    std::vector<double> v, pr;
    CSR_matrix b(N);

    generate_matrix(N, 10 + seed, b);
    generate_vector(N, 1+seed, v);

    product(b, v, pr);

    std::cout << "Matrix-vector product test\n";
    if (debug) {
        PRINT_VECTOR(b.IA, "IA");
        PRINT_VECTOR(b.JA, "JA");
        PRINT_VECTOR(b.values, "A ");

        std::cout << "[\n";
        for (int i = 0; i < N; ++i) {
            std::cout << " [";
            for (int j = 0; j < N; ++j) {
                std::cout << b.getv(i, j) << ",  ";
            }
            std::cout << "],\n";
        }
        std::cout << "]\n";

        PRINT_VECTOR(v, "v  ");
        PRINT_VECTOR(pr, "A v");
    }
    PRINT_VECTOR(control_sum(pr, cs_seed), "control sum of A v");
    std::cout << std::endl;
}


void test_linear_combination(
        int N, uint_fast64_t seed,
        uint_fast64_t cs_seed, bool debug) {

    std::vector<double> v, h;
    generate_vector(N, 1+seed, v);
    generate_vector(N, 2+seed, h);

    std::cout << "Linear combination test\n";
    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }
    linear_combination(3.14, 2.78, h, v);
    if (debug) {
        PRINT_VECTOR(h, "h = 3.14*h + 2.78*v");
    }
    PRINT_VECTOR(control_sum(h, cs_seed), "control sum of h");
    std::cout << std::endl;
}


void test_dot_product(int N, uint_fast64_t seed, bool debug) {
    std::cout << "Dot product test\n";
    std::vector<double> v, h;
    generate_vector(N, 1+seed, v);
    generate_vector(N, 2+seed, h);

    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }
    std::cout << "dot(h, v) = " << dot_product(h, v) << "\n\n";
}

#endif