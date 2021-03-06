#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include <mpi.h>

#include "mpi_operations.h"
#include "mpi_timetest.h"
#include "mpi_test.h"

#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

#define HANDLE_ARG(name) else if (arg.find("--" #name "=") == 0) \
    { name = std::stoi(arg.substr(3 + std::strlen(#name))); }

int main(int argc, char **args) {
    int process_cnt, my_id;
    // init, get process count and id ot current process
    handle_res(MPI_Init(&argc, &args));
    handle_res(MPI_Comm_size(MPI_COMM_WORLD, &process_cnt));
    handle_res(MPI_Comm_rank(MPI_COMM_WORLD, &my_id));


    int n, rep = 100, num_threads = 1, seed = 1, cs_seed = 2,
        nx = 0, ny = 0, px = 0, py = 0;
    bool debug = false, timetest = false;
    if (argc == 1) {
        if (!my_id) {
            std::cout << "\nUsage: " << args[0]
                << " [--timetest] [--rep=REPEATS] [--debug] [--seed=SEED] [--cs_seed=CS_SEED] "
                "[--num_threads=NUM_THREADS] [--nx=NX] [--ny=NY] [--px=PX] [--py=PY]\n"
                "  --timetest  measure mean time of operations\n"
                "  --rep=REPEATS  set repeats for averaging time\n"
                "  --debug  print matrices and vectors to stdout\n"
                "  --seed=SEED  set seed for generating data\n"
                "  --cs_seed=CS_SEED  set seed for control sums\n"
                "  --num_threads=NUM_THREADS set number of threads for OpenMP\n"
                "  --nx=NX --ny=NY set nx and ny (nx*ny = n)\n"
                "  --px=PX --py=PY set px and py (px*py = number of MPI processes)\n";

            std::cout << std::endl;
        }

        goto main_exit;
    }

    // handle all args
    for (int i = 0; i < argc; ++i) {
        std::string arg(args[i]);
        if (arg == "--debug") {
            debug = true;
        }
        else if (arg.find("--num_threads=") == 0) {
            num_threads = std::stoi(arg.substr(14));
        }
        else if (arg == "--timetest") {
            timetest = true;
        }
        HANDLE_ARG(cs_seed)
        HANDLE_ARG(seed)
        HANDLE_ARG(rep)
        HANDLE_ARG(nx)
        HANDLE_ARG(ny)
        HANDLE_ARG(px)
        HANDLE_ARG(py)
    }
    omp_set_num_threads(num_threads);

    n = nx * ny;
    // px*py should be equal process count
    if (nx < 1 || ny < 1 || px < 1 || py < 1 || num_threads < 1 || rep < 1 || nx < px || ny < py) {
        std::cerr << "Bad input\n";
        goto main_exit;
    }
    if (px * py != process_cnt) {
        std::cerr << "px*py != process_cnt" << std::endl;
        std::cerr << "[[[" << nx << " " << ny << " " << n << "]]]" << std::endl;
        std::cerr << "[[[" << px << " " << py << " " << process_cnt << "]]]" << std::endl;
        return 111;
    }


    char processor_name[256];
    int processor_name_len;
    handle_res(MPI_Get_processor_name(processor_name, &processor_name_len));

    if (!my_id) {
        std::cout << "Number of processes: " << process_cnt
            << "\nMy id: " << my_id
            << "\nProcessor name: " << processor_name << std::endl;
        std::cout << "control sum is [sum, L2, min, max, dot(x, random vector)]" << std::endl;
    }

    std::cout.precision(16);

    if (!timetest) {
        test_dot_product_mpi(n, nx, ny, px, py, seed, my_id, process_cnt, debug);
        test_mat_vec_product_mpi(n, nx, ny, px, py, seed, cs_seed, my_id, process_cnt, debug);
        test_linear_combination_mpi(n, nx, ny, px, py, seed, cs_seed, my_id, process_cnt, debug);
    }
    else {
        int printer = 0;
        std::vector<double> csum;
        size_t a, b;
        double t;
        t = avg_time_of_mat_vec_product_mpi(n, nx, ny, px, py, rep, seed, cs_seed, csum,
            a, b, my_id, process_cnt);
        if (my_id == printer) {
            // print time and data bandwidth
            std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
                << ": " << t << "\n  " << a / t / 1e9 << ".." << b / t / 1e9 << " GB/s\n";
            PRINT_VECTOR(csum, "  control sum");
        }

        double dot = -1;
        t = avg_time_of_dot_product_mpi(n, nx, ny, px, py, rep, seed, dot, a, my_id, process_cnt);
        if (my_id == printer) {
            // print time and data bandwidth
            std::cout << "\n\navg time of dot product, n=" << n << ", repeats=" << rep
                << ": " << t << std::endl
                << "  " << a / t / 1e9 << " GB/s\n"
                << "  dot : " << dot << std::endl;
        }


        t = avg_time_of_linear_combination_mpi(n, nx, ny, px, py, rep, seed,
            cs_seed, csum, a, my_id, process_cnt);
        if (my_id == printer) {
            // print time and data bandwidth
            std::cout << "\n\navg time of linear combination, n=" << n << ", repeats=" << rep
                << ": " << t << std::endl
                << "  " << a / t / 1e9 << " GB/s\n";
            PRINT_VECTOR(csum, "  control sum");
        }

    }

main_exit:
    MPI_Finalize();
}
