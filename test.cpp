#ifndef TEST_CPP
#define TEST_CPP

#include "test.h"


void generate_matrix(size_t n, uint_fast64_t seed, CSR_matrix &ans) {
    ans.set_size(n);
    double x = std::sin(seed) * 1234.56789;
    for (size_t i = 0; i < n; ++i) {
        ans.IA[i] = ans.JA.size();
        for (size_t j = (i-1) * (i>0); j < n && j < i+2; ++j) {
            ans.JA.push_back(j);
            ans.values.push_back(std::sin(x += 10));
        }
    }
}


void generate_vector(
        size_t n, uint_fast64_t seed,
        std::vector<double> &ans) {

    double x = std::sin(seed) * 1234.56789;

    ans.resize(n, 0);
    for (size_t i = 0; i < n; ++i) {
        ans[i] = std::sin(x += 10);
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
        size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed, bool debug) {
    std::vector<double> v, pr;
    CSR_matrix b(n);

    generate_matrix(n, 10 + seed, b);
    generate_vector(n, 1+seed, v);

    product(b, v, pr);

    std::cout << "Matrix-vector product test\n";
    if (debug) {
        PRINT_VECTOR(b.IA, "IA");
        PRINT_VECTOR(b.JA, "JA");
        PRINT_VECTOR(b.values, "A ");

        std::cout << "[\n";
        for (size_t i = 0; i < n; ++i) {
            std::cout << " [";
            for (size_t j = 0; j < n; ++j) {
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
        size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed, bool debug) {

    std::vector<double> v, h;
    generate_vector(n, 1+seed, v);
    generate_vector(n, 2+seed, h);

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


void test_dot_product(size_t n, uint_fast64_t seed, bool debug) {
    std::cout << "Dot product test\n";
    std::vector<double> v, h;
    generate_vector(n, 1+seed, v);
    generate_vector(n, 2+seed, h);

    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }
    std::cout << "dot(h, v) = " << dot_product(h, v) << "\n\n";
}

#endif
