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
#include <utility>

#include <mpi.h>

#include "csr_matrix.h"


#define handle_res(res) handle_res_f(res, #res " FAILED")
void handle_res_f(int res, const char *err_msg);

size_t my_begin(size_t n, size_t my_id, size_t proc_cnt);
size_t my_end(size_t n, size_t my_id, size_t proc_cnt);


double dot_product(
    const std::vector<double> &a,
    const std::vector<double> &b);

void linear_combination(double a, double b,
    std::vector<double> &x, const std::vector<double> &y);


struct comm_data {
    std::vector<size_t> l2g, g2l, part, send_list, recv_list,
        send_offset, recv_offset;
    size_t n, n_own, my_id, proc_cnt;

    comm_data(size_t n, size_t my_id, size_t proc_cnt)
        : n(n), my_id(my_id), proc_cnt(proc_cnt) {}
};

std::vector<double> control_sum_mpi(comm_data &cd,
    const std::vector<double> &x,
    int seed);

double random(int n, int seed); // (0 : 1]
double normal(int n, int seed); // return random value from N(0, 1)

// uses only cd.l2g, cd.n_own
void generate_matrix(comm_data &cd,
        CSR_matrix &mat_piece,
        size_t nx,
        int seed);

// uses only cd.l2g, cd.n_own
void generate_vector(comm_data &cd,
        std::vector<double> &vec_piece,
        int seed);


// cd.proc_cnt should be equal to px * py
// cd.n should be equal to nx * ny
void init_l2g_part(comm_data &cd,
        size_t nx, size_t ny, size_t px, size_t py);
void init(comm_data &cd,
        size_t nx, size_t ny, size_t px, size_t py,
        int seed,
        CSR_matrix &mat_piece,
        std::vector<double> &vec_piece);

void update(comm_data &cd,
    std::vector<double> &vec_piece);

// ans_piece should be resized to mat_piece.size() with zeros
void product(comm_data &cd,
        CSR_matrix &mat_piece,
        std::vector<double> &vec_piece,
        std::vector<double> &ans_piece);

#endif
