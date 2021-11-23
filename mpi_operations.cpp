#ifndef MPI_OPERATIONS_CPP
#define MPI_OPERATIONS_CPP

#include "mpi_operations.h"
#include "my_time.h"


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
    // first n % proc_cnt id`s have (1 + n / proc_cnt) elements each
    // other have n / proc_cnt elements each
    return my_id * (n / proc_cnt)
        + (my_id < n % proc_cnt ? my_id : n % proc_cnt);
}

size_t my_end(size_t n, size_t my_id, size_t proc_cnt) {
    return my_begin(n, my_id+1, proc_cnt);
}

size_t get_id(size_t i, size_t n, size_t proc_cnt) {
    // my_begin(n, id, proc_cnt) <= i < my_end(n, id, proc_cnt)
    auto c = n % proc_cnt;
    auto id = i / (1 + n / proc_cnt); // it is id assuming that id<c
    // if out assumption is wrong
    if (id >= c) {
        id = c / (1 + n / proc_cnt)
            + (i-c) / (n / proc_cnt);
    }
    return id;
}

//

double dot_product(
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


std::vector<double> control_sum_mpi(comm_data &cd,
        const std::vector<double> &x,
        int seed) {
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
    generate_vector(cd, h, seed);
    all_ans.push_back(dot_product(h, x));
    return all_ans;
}


// hash from stackoverflow
unsigned f_s(unsigned a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

// in (0:1]
double random(size_t n, size_t seed) {
    unsigned ans = f_s(f_s(f_s(n) ^ f_s(seed)) ^ f_s(f_s(n >> 32) ^ f_s(seed >> 32)));
    return 1 - ans / double(1ll << 32);
}


// Box-Muller transform
double normal(size_t n, size_t seed) {
    static const double pi = std::atan2(0, -1);
    auto u1 = random(2*(n>>1), seed),
        u2 = random(2*(n>>1)+1, seed);
    if (n & 1) {
        return std::sqrt(-2 * std::log(u1)) * std::cos(2*pi*u2);
    }
    return std::sqrt(-2 * std::log(u1)) * std::sin(2*pi*u2);
}


// uses only cd.l2g, cd.n_own
void generate_matrix(comm_data &cd,
        CSR_matrix &mat_piece,
        size_t nx,
        int seed) {

    mat_piece.JA.reserve(5*cd.n_own);
    mat_piece.values.reserve(5*cd.n_own);
    mat_piece.set_size(cd.n_own);
    for (size_t i = 0; i < cd.n_own; ++i) {
        mat_piece.IA[i] = mat_piece.JA.size();
        auto gi = cd.l2g[i];
        for (size_t j : {gi-nx, gi-1, gi, gi+1, gi+nx}) {
            // assuming unsignedness of j to avoid comparison with zero
            // if true value of gi-nx is <0 then unsigned value is very big
            // (if gi is < (1 << (sizeof(size_t) * CHAR_BIT - 1)))
            if (j >= cd.n) {
                continue;
            }
            mat_piece.JA.push_back(j);
            mat_piece.values.push_back(normal(gi * cd.n + j, seed));
        }
    }
}


// uses only cd.l2g, cd.n_own
void generate_vector(comm_data &cd,
        std::vector<double> &vec_piece,
        int seed) {

    vec_piece.resize(cd.n_own);
    #pragma omp parallel for
    for (size_t i = 0; i < cd.n_own; ++i) {
        vec_piece[i] = normal(cd.l2g[i], seed);
    }
}


// cd.proc_cnt should be equal to px * py
// cd.n should be equal to nx * ny
void init_l2g_part(comm_data &cd,
        size_t nx, size_t ny, size_t px, size_t py) {

    // go through my indices in grid and add them in l2g
    for (size_t i = my_begin(nx, cd.my_id / py, px);
            i < my_end(nx, cd.my_id / py, px); ++i) {
        for (size_t j = my_begin(ny, cd.my_id % py, py);
                j < my_end(ny, cd.my_id % py, py); ++j) {
            cd.l2g.push_back(i * ny + j);
        }
    }
    cd.n_own = cd.l2g.size();

    // part[i] -- id of process which own i-th element of vector
    cd.part.resize(cd.n);
    for(size_t id = 0; id < cd.proc_cnt; ++id) {
        for (size_t i = my_begin(nx, id / py, px);
                i < my_end(nx, id / py, px); ++i) {
            for (size_t j = my_begin(ny, id % py, py);
                    j < my_end(ny, id % py, py); ++j) {
                cd.part[i * ny + j] = id;
            }
        }
    }
}


// cd.proc_cnt should be equal to px * py
// cd.n should be equal to nx * ny
void init(comm_data &cd,
        size_t nx, size_t ny, size_t px, size_t py,
        int seed,
        CSR_matrix &mat_piece,
        std::vector<double> &vec_piece) {

    init_l2g_part(cd, nx, ny, px, py);

    generate_matrix(cd, mat_piece, nx, seed);
    generate_vector(cd, vec_piece, seed*2+1);

    // create lists of indices for sending and receiving
    // send_list[i] -- indices to send to i-th process
    // recv_list[i] -- indices to receive from i-th process
    std::vector<std::vector<size_t> > send_list(cd.proc_cnt), recv_list(cd.proc_cnt);
    for (size_t i = 0; i < cd.n_own; ++i) {
        for (size_t k = mat_piece.JA_begin(i); k < mat_piece.JA_end(i); ++k) {
            auto col = mat_piece.JA[k];
            if (cd.part[col] != cd.my_id) {
                cd.l2g.push_back(col);
                recv_list[cd.part[col]].push_back(col);
                send_list[cd.part[col]].push_back(cd.l2g[i]);
            }
        }
    }

    // sort, unique and flatten send_list
    for (auto &x : send_list) {
        cd.send_offset.push_back(cd.send_list.size());
        std::sort(x.begin(), x.end());
        x.erase(std::unique(x.begin(), x.end()), x.end());
        for (auto y : x) {
            cd.send_list.push_back(y);
        }
    }
    cd.send_offset.push_back(cd.send_list.size());
    // sort, unique and flatten recv_list
    for (auto &x : recv_list) {
        cd.recv_offset.push_back(cd.recv_list.size());
        std::sort(x.begin(), x.end());
        x.erase(std::unique(x.begin(), x.end()), x.end());
        for (auto y : x) {
            cd.recv_list.push_back(y);
        }
    }
    cd.recv_offset.push_back(cd.recv_list.size());

    cd.g2l.resize(cd.n, -(intptr_t)&cd);
    #pragma omp parallel for
    for (size_t i = 0; i < cd.l2g.size(); ++i) {
        cd.g2l[cd.l2g[i]] = i;
    }
}

void update(comm_data &cd,
    std::vector<double> &vec_piece) {

    std::vector<double> send_buf(cd.send_list.size()),
        recv_buf(cd.recv_list.size());

    // fill send_buf with data from our vector
    #pragma omp parallel for
    for (size_t i = 0; i < send_buf.size(); ++i) {
        send_buf[i] = vec_piece[cd.g2l[cd.send_list[i]]];
    }

    // send and receive from/to neighbours
    std::vector<MPI_Request> reqs(2 * cd.proc_cnt, MPI_REQUEST_NULL);
    for (size_t id = 0; id < cd.proc_cnt; ++id) {
        if (id != cd.my_id) {
            if (cd.send_offset[id+1] > cd.send_offset[id]) {
                handle_res(MPI_Isend(&send_buf[cd.send_offset[id]],
                    cd.send_offset[id+1] - cd.send_offset[id], MPI_DOUBLE,
                    id, id, MPI_COMM_WORLD, &reqs[2*id]));
            }
            if (cd.recv_offset[id+1] > cd.recv_offset[id]) {
                handle_res(MPI_Irecv(&recv_buf[cd.recv_offset[id]],
                    cd.recv_offset[id+1] - cd.recv_offset[id], MPI_DOUBLE,
                    id, cd.my_id, MPI_COMM_WORLD, &reqs[2*id+1]));
            }
        }
    }
    handle_res(MPI_Waitall(reqs.size(), &reqs[0], MPI_STATUSES_IGNORE));

    if (vec_piece.size() != cd.l2g.size()) {
        vec_piece.resize(cd.l2g.size());
    }
    // fill vec_piece with updated values
    #pragma omp parallel for
    for (size_t i = 0; i < recv_buf.size(); ++i) {
        vec_piece[cd.g2l[cd.recv_list[i]]] = recv_buf[i];
    }
}


// ans_piece should be resized to mat_piece.size() with zeros
void product(comm_data &cd,
        CSR_matrix &mat_piece,
        std::vector<double> &vec_piece,
        std::vector<double> &ans_piece) {

    #pragma omp parallel for
    for (size_t i = 0; i < mat_piece.size(); ++i) {
        double sum = 0;
        for (size_t k = mat_piece.JA_begin(i); k < mat_piece.JA_end(i); ++k) {
            size_t col = mat_piece.JA[k];
            sum += mat_piece.values[k] * vec_piece[cd.g2l[col]];
        }
        ans_piece[i] = sum;
    }
}

#endif
