#ifndef TIMETEST_H
#define TIMETEST_H

#include <vector>
#include <chrono>

#include "matrices.h"
#include "test.h"


double time();

double avg_time_of_mat_vec_product(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum,
        size_t &min_mem_usage, size_t &max_mem_usage);

double avg_time_of_linear_combination(size_t n, size_t repeats,
        uint_fast64_t seed, uint_fast64_t cs_seed,
        std::vector<double> &csum, size_t &mem_usage);

double avg_time_of_dot_product(size_t n, size_t repeats,
        uint_fast64_t seed, double &dot, size_t &mem_usage);

#endif
