#ifndef TEST_H
#define TEST_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "matrices.h"


void generate_matrix(size_t n, uint_fast64_t seed, CSR_matrix &ans);
void generate_vector(size_t n, uint_fast64_t seed, std::vector<double> &ans);

std::vector<double> control_sum(const std::vector<double> &x, uint_fast64_t seed);

void test_mat_vec_product(size_t n, uint_fast64_t seed,
    uint_fast64_t cs_seed, bool debug);

void test_linear_combination(size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed, bool debug);

void test_dot_product(size_t n, uint_fast64_t seed, bool debug);

#endif
