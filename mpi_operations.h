#ifndef MPI_OPERATIONS_H
#define MPI_OPERATIONS_H

#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define my_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "what is happening here?"
#endif


#include <cmath>
#include <algorithm>
#include <vector>

#include <chrono>
#include <thread>
#include <iostream>

#include <mpi.h>

#include "matrices.h"


#define handle_res(res) handle_res_f(res, #res " FAILED")

void handle_res_f(int res, const char *err_msg);

size_t my_begin(size_t n, size_t my_id, size_t proc_cnt);
size_t my_end(size_t n, size_t my_id, size_t proc_cnt);


void generate_matrix_mpi(size_t n, uint_fast64_t seed, CSR_matrix &my_piece,
    size_t my_id, size_t proc_cnt);

void generate_vector_mpi(size_t n, uint_fast64_t seed,
    std::vector<double> &ans, size_t my_id, size_t proc_cnt);


double dot_product_mpi(const std::vector<double> &a,
    const std::vector<double> &b);

void test_dot_product_mpi(size_t n, uint_fast64_t seed,
        size_t my_id, size_t proc_cnt, bool debug);


std::vector<double> control_sum_mpi(size_t n,
        const std::vector<double> &x,
        uint_fast64_t seed, size_t my_id, size_t proc_cnt);

void test_linear_combination_mpi(
        size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed,
        size_t my_id, size_t proc_cnt, bool debug);


void product_mpi(size_t n,
        const CSR_matrix &mat_piece,
        const std::vector<double> &v_piece,
        std::vector<double> &ans_piece,
        size_t my_id, size_t proc_cnt);

void test_mat_vec_product_mpi(
        size_t n, uint_fast64_t seed, uint_fast64_t cs_seed,
        size_t my_id, size_t proc_cnt, bool debug);

#endif
