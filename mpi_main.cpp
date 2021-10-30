#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_operations.h"
#include "mpi_timetest.h"

#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)


int main(int argc, char **args) {
    int process_cnt, my_id;
    handle_res(MPI_Init(&argc, &args));
    handle_res(MPI_Comm_size(MPI_COMM_WORLD, &process_cnt));
    handle_res(MPI_Comm_rank(MPI_COMM_WORLD, &my_id));


    int n = 1000, rep = 100, /* num_threads = 8,*/ seed = 1, cs_seed = 2;
    bool debug = false, timetest = false;
    if (argc == 1) {
        if (!my_id) {
            std::cout << "Usage: " << args[0]
                << " n [--rep=REPEATS] [--seed=SEED] [--timetest]"
                << " [--cs_seed=CS_SEED] [--debug]\n\n"
                << "  --timetest  measure mean time of operations\n"
                << "  --rep=REPEATS  set repeats for averaging time\n"
                << "  --debug  print matrices and vectors to stdout\n"
                << "  --seed=SEED  set seed for generating data\n"
                << "  --cs_seed=CS_SEED  set seed for control sums\n";

            std::cout << std::endl;
        }

        goto main_exit;
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg(args[i]);
        if (arg == "--debug") {
            debug = true;
        }
        else if (arg.find("--cs_seed=") == 0) {
            cs_seed = std::stoi(arg.substr(10));
        }
        else if (arg.find("--seed=") == 0) {
            seed = std::stoi(arg.substr(7));
        }
        else if (arg == "--timetest") {
            timetest = true;
        }
        else if (arg.find("--rep=") == 0) {
            rep = std::stoi(arg.substr(6));
        }
    }

    if (argc > 1) {
        n = std::stoi(args[1]);
    }


    char processor_name[256];
    int processor_name_len;
    handle_res(MPI_Get_processor_name(processor_name, &processor_name_len));

    if (!my_id) {
        std::cout << "Number of processes: " << process_cnt
            << "\nMy id: " << my_id
            << "\nProcessor name: " << processor_name << std::endl;
    }

    std::cout.precision(16);

    if (!timetest) {
        test_dot_product_mpi(n, seed, my_id, process_cnt, debug);
        test_mat_vec_product_mpi(n, seed, cs_seed, my_id, process_cnt, debug);
        test_linear_combination_mpi(n, seed, cs_seed, my_id, process_cnt, debug);
    }
    else {
        std::vector<double> csum;
        size_t a, b;
        double t;
        t = avg_time_of_mat_vec_product_mpi(n, rep, seed, cs_seed, csum,
            a, b, my_id, process_cnt);
        if (!my_id) {
            std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
                << ": " << t << "\n  " << a / t / 1e9 << ".." << b / t / 1e9 << " GB/s\n";
            PRINT_VECTOR(csum, "  control sum");
        }

        t = avg_time_of_linear_combination_mpi(n, rep, seed, cs_seed, csum, a, my_id, process_cnt);
        if (!my_id) {
            std::cout << "\n\navg time of linear combination, n=" << n << ", repeats=" << rep
                << ": " << t << std::endl
                << "  " << a / t / 1e9 << " GB/s\n";
            PRINT_VECTOR(csum, "  control sum");
        }

        double dot = -1;
        t = avg_time_of_dot_product_mpi(n, rep, seed, dot, a, my_id, process_cnt);
        if (!my_id) {
            std::cout << "\n\navg time of dot product, n=" << n << ", repeats=" << rep
                << ": " << t << std::endl
                << "  " << a / t / 1e9 << " GB/s\n"
                << "  dot : " << dot << std::endl;
        }
    }

main_exit:
    MPI_Finalize();
}
