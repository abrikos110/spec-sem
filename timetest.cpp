#ifndef TIMETEST_CPP
#define TIMETEST_CPP

#include "timetest.h"


double time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        / 1e9;
}


double avg_time_of_mat_vec_product(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum,
        size_t &min_mem_usage, size_t &max_mem_usage) {

    std::vector<double> v, pr;
    CSR_matrix a;

    generate_vector(n, 1 + seed, v);
    generate_matrix(n, 10 + seed, a);

    // assuming a.JA.size() > n
    min_mem_usage = a.JA.size() * (sizeof(size_t) + sizeof(double)) + n * 2 * sizeof(double);
    max_mem_usage = a.JA.size() * (sizeof(size_t) + sizeof(double) * 2) + n * sizeof(double);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        product(a, v, pr);
    }
    st = time() - st;

    csum = control_sum(pr, cs_seed);

    return st / repeats;
}


double avg_time_of_linear_combination(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum, size_t &mem_usage) {

    // control sums are different with different repeats
    std::vector<double> x, y;
    generate_vector(n, 1+seed, x);
    generate_vector(n, 2+seed, y);

    mem_usage = 3 * n * sizeof(double);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        linear_combination(0.72, 0.27, x, y); // this is cumulative sum for my laziness
        // sum of coeffs less than 1 to avoid exponential growth
    }
    st = time() - st;

    csum = control_sum(x, cs_seed);

    return st / repeats;
}


double avg_time_of_dot_product(size_t n, size_t repeats,
        uint_fast64_t seed, double &dot, size_t &mem_usage) {

    std::vector<double> x, y;
    generate_vector(n, 1+seed, x);
    generate_vector(n, 2+seed, y);

    mem_usage = 2 * n * sizeof(double);

    double st = time();
    for (size_t i = 0; i < repeats; ++i) {
        dot = dot_product(x, y);
    }
    st = time() - st;

    return st / repeats;
}

#endif
