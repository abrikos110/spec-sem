#ifndef MPI_OPERATIONS_CPP
#define MPI_OPERATIONS_CPP

#include <cmath>
#include <algorithm>
#include <vector>

#include "matrices.cpp"

size_t my_begin(size_t n, size_t my_id, size_t proc_cnt) {
    return my_id * (n / proc_cnt)
        + (my_id < n % proc_cnt ? my_id : n % proc_cnt);
}

size_t my_end(size_t n, size_t my_id, size_t proc_cnt) {
    return my_begin(n, my_id+1, proc_cnt);
}


void generate_matrix_mpi(size_t n, uint_fast64_t seed, CSR_matrix &my_piece,
        size_t my_id, size_t proc_cnt) {

    double x = std::sin(seed) * 1234.56789
        + 10 * (my_id > 0) * (-1 + 3 * my_begin(n, my_id, proc_cnt));
    my_piece.set_size(my_end(n, my_id, proc_cnt) - my_begin(n, my_id, proc_cnt));

    for (size_t i = my_begin(n, my_id, proc_cnt);
            i < my_end(n, my_id, proc_cnt); ++i) {
        my_piece.IA[i] = my_piece.JA.size();
        for (size_t j = (i-1) * (i>0); j < n && j < i+2; ++j) {
            my_piece.JA.push_back(j);
            my_piece.values.push_back(std::sin(x += 10));
        }
    }

}


void generate_vector_mpi(
        size_t n, uint_fast64_t seed,
        std::vector<double> &ans,
        size_t my_id, size_t proc_cnt) {

    double x = std::sin(seed) * 1234.56789
        + 10 * my_begin(n, my_id, proc_cnt);

    ans.reserve(my_end(n, my_id, proc_cnt) - my_begin(n, my_id, proc_cnt));
    for (size_t i = my_begin(n, my_id, proc_cnt);
            i < my_end(n, my_id, proc_cnt); ++i) {
        ans.push_back(std::sin(x += 10));
    }
}


double dot_product_mpi(
    const std::vector<double> &a,
    const std::vector<double> &b) {

    double ans = 0, all_ans = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    MPI_Allreduce(&ans, &all_ans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return all_ans;
}


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

void test_dot_product_mpi(int n, uint_fast64_t seed,
        size_t my_id, size_t proc_cnt, bool debug) {

    if (!my_id) {
        std::cout << "MPI Dot product test\n";
    }
    std::vector<double> v, h;
    generate_vector_mpi(n, 1+seed, v, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, h, my_id, proc_cnt);

    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }

    double dot = dot_product_mpi(h, v);
    if (!my_id) {
        std::cout << "dot(h, v) = " << dot << "\n\n";
    }
}

/////////////////////////////////////////////////////////

std::vector<double> control_sum_mpi(size_t n,
        const std::vector<double> &x,
        uint_fast64_t seed, size_t my_id, size_t proc_cnt) {
    // sum, L2, min, max, dot(x, random vector)
    std::vector<double> ans, all_ans;
    all_ans.resize(4, 0);

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
    MPI_Allreduce(&ans[0], &all_ans[0], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ans[1], &all_ans[1], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&ans[2], &all_ans[2], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ans[3], &all_ans[3], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    all_ans[1] = std::sqrt(all_ans[1]);

    std::vector<double> h;
    generate_vector_mpi(n, seed, h, my_id, proc_cnt);
    all_ans.push_back(dot_product_mpi(h, x));
    return all_ans;
}


void test_linear_combination_mpi(
        size_t n, uint_fast64_t seed,
        uint_fast64_t cs_seed,
        size_t my_id, size_t proc_cnt, bool debug) {

    std::vector<double> v, h;
    generate_vector_mpi(n, 1+seed, v, my_id, proc_cnt);
    generate_vector_mpi(n, 2+seed, h, my_id, proc_cnt);

    if (!my_id) {
        std::cout << "MPI Linear combination test\n";
    }
    if (debug) {
        PRINT_VECTOR(v, "v");
        PRINT_VECTOR(h, "h");
    }
    linear_combination(3.14, 2.78, h, v);
    if (debug) {
        PRINT_VECTOR(h, "h = 3.14*h + 2.78*v");
    }
    auto cs = control_sum_mpi(n, h, cs_seed, my_id, proc_cnt);
    if (!my_id) {
        PRINT_VECTOR(cs, "control sum of h");
        std::cout << std::endl;
    }
}

///////////////////////////////////////////
// NOT COMPLETED
void product_mpi(size_t n,
        const CSR_matrix &mat_piece,
        const std::vector<double> &v_piece,
        std::vector<double> &ans_piece,
        size_t my_id, size_t proc_cnt) {

    size_t my_len = my_end(n, my_id, proc_cnt) - my_begin(n, my_id, proc_cnt);
    ans_piece.resize(my_len, 0);

    std::vector<size_t> want;
    for (size_t i = 0; i < mat_piece.size(); ++i) {
        for (size_t k = mat_piece.JA_begin(i); k < mat_piece.JA_end(i); ++k) {
            size_t col = mat_piece.JA[k];
            if (col < my_begin(n, my_id, proc_cnt) || col >= my_end(n, my_id, proc_cnt)) {
                want.push_back(col);
            }
        }
    }
    want.erase(std::unique(want.begin(), want.end()), want.end());

    std::vector<size_t> sizes;
    for (size_t i = 0, li = 0; i <= want.size(); ++i) {
        if (/*??*/i >= my_end(n, sizes.size(), proc_cnt)) {
            sizes.push_back(i - li);
            li = i;
        }
    }

}

#endif
