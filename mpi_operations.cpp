#ifndef MPI_OPERATIONS_CPP
#define MPI_OPERATIONS_CPP

#include "mpi_operations.h"


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

void handle_res_f(int res, const char *err_msg) {
    if (res != MPI_SUCCESS) {
        std::cerr << err_msg << " " << res << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
}


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
        my_piece.IA[i - my_begin(n, my_id, proc_cnt)] = my_piece.JA.size();
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

    size_t ib = my_begin(n, my_id, proc_cnt);
    double x = std::sin(seed) * 1234.56789;

    ans.resize(my_end(n, my_id, proc_cnt) - my_begin(n, my_id, proc_cnt));
    #pragma omp parallel for
    for (size_t i = ib; i < my_end(n, my_id, proc_cnt); ++i) {
        ans[i - ib] = std::sin(x + 10 * (i+1));
    }
}


double dot_product_mpi(
        const std::vector<double> &a,
        const std::vector<double> &b) {

    double ans = 0, all_ans = 0;
    #pragma omp parallel for reduction(+:ans)
    for (size_t i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    handle_res(MPI_Allreduce(&ans, &all_ans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

    return all_ans;
}


void linear_combination(double a, double b,
        std::vector<double> &x, const std::vector<double> &y) {
    // x = a * x + b * y
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = a * x[i] + b * y[i];
    }
}


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

std::vector<double> control_sum_mpi(size_t n,
        const std::vector<double> &x,
        uint_fast64_t seed, size_t my_id, size_t proc_cnt) {
    // sum, L2, min, max, dot(x, random vector)
    std::vector<double> ans, all_ans;

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
    all_ans = ans;
    handle_res(MPI_Allreduce(&ans[0], &all_ans[0], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    handle_res(MPI_Allreduce(&ans[1], &all_ans[1], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

    handle_res(MPI_Allreduce(&ans[2], &all_ans[2], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    handle_res(MPI_Allreduce(&ans[3], &all_ans[3], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    all_ans[1] = std::sqrt(all_ans[1]);

    std::vector<double> h;
    generate_vector_mpi(n, seed, h, my_id, proc_cnt);
    all_ans.push_back(dot_product_mpi(h, x));
    return all_ans;
}


void init(size_t n, size_t &n_own,
        const CSR_matrix &mat_piece,
        std::vector<size_t> &l2g,
        std::vector<size_t> &g2l,
        std::vector<size_t> &part,
        std::vector<std::pair<size_t, size_t> > &ask,
        size_t my_id, size_t proc_cnt) {

    // v_piece[: n_own] is owned data
    // v_piece[n_own :] is data from other processes
    n_own = my_end(n, my_id, proc_cnt) - my_begin(n, my_id, proc_cnt);
    for (size_t i = my_begin(n, my_id, proc_cnt);
            i < my_end(n, my_id, proc_cnt); ++i) {
        l2g.push_back(i);
    }
    // part[i] is id of owner of i-th element
    for (size_t i = 0; i < proc_cnt; ++i) {
        part.resize(part.size() + my_end(n, i, proc_cnt) - my_begin(n, i, proc_cnt), i);
    }

    // check for every nonzero element of matrix
    // and add their indices for future transfers between processes
    for (size_t i = 0; i < mat_piece.size(); ++i) {
        for (size_t k = mat_piece.JA_begin(i); k < mat_piece.JA_end(i); ++k) {
            size_t col = mat_piece.JA[k];
            if (part[col] != my_id) {
                l2g.push_back(col);
                ask.emplace_back(l2g[i], col);
            }
        }
    }
    l2g.erase(std::unique(l2g.begin() + n_own, l2g.end()), l2g.end());
    ask.erase(std::unique(ask.begin(), ask.end()), ask.end());

    g2l.resize(n, ((size_t)1) << (8 * sizeof(size_t) - 4));
    #pragma omp parallel for
    for (size_t i = 0; i < l2g.size(); ++i) {
        g2l[l2g[i]] = i;
    }
}

void update(size_t n_own,
        std::vector<double> &v_piece,
        const std::vector<size_t> &l2g,
        const std::vector<size_t> &g2l,
        const std::vector<size_t> &part,
        const std::vector<std::pair<size_t, size_t> > &ask,
        size_t my_id, size_t proc_cnt) {

    std::vector<double> asked;
    v_piece.resize(l2g.size());
    // fill vector with data owned by process
    for (auto k : ask) {
        asked.push_back(v_piece[g2l[k.first]]);
    }

    std::vector<MPI_Request> reqs(proc_cnt * 2, MPI_REQUEST_NULL);
    for (size_t i = n_own, li = n_own, id = 0; i <= l2g.size(); ++i) {
        // find segment of indices with owner id `id` and receive data
        while (i == l2g.size() || part[l2g[i]] != id) {
            if (i - li > 0) {
                handle_res(MPI_Irecv(&v_piece[li], i - li, MPI_DOUBLE,
                    id, my_id, MPI_COMM_WORLD, &reqs[id]));
            }
            li = i; ++id;
            if (i == l2g.size()) {
                break;
            }
        }
    }
    for (size_t i = 0, li = 0, id = 0; i <= ask.size(); ++i) {
        // find segment of indices with receiver id `id` and send data
        while (i == ask.size() || part[ask[i].second] != id) {
            if (i - li > 0) {
                handle_res(MPI_Isend(&asked[li], i - li, MPI_DOUBLE,
                    id, id, MPI_COMM_WORLD, &reqs[id + proc_cnt]));
            }
            li = i; ++id;
            if (i == ask.size()) {
                break;
            }
        }
    }
    handle_res(MPI_Waitall(reqs.size(), &reqs[0], MPI_STATUSES_IGNORE));
}


// ans_piece should be resized to mat_piece.size() with zeros
void product_mpi(const CSR_matrix &mat_piece,
        std::vector<double> &v_piece,
        std::vector<double> &ans_piece,
        const std::vector<size_t> &g2l) {

    #pragma omp parallel for
    for (size_t i = 0; i < mat_piece.size(); ++i) {
        double sum = 0;
        for (size_t k = mat_piece.JA_begin(i); k < mat_piece.JA_end(i); ++k) {
            size_t col = mat_piece.JA[k];
            sum += mat_piece.values[k] * v_piece[g2l[col]];
        }
        ans_piece[i] = sum;
    }
}

#endif
