#include <iostream>
#include <string>
#include <omp.h>

#include "matrices.h"
#include "test.h"
#include "timetest.h"


#define PRINT_VECTOR(x, name) do {std::cout << (name) << " : [";\
    for (auto iggg : (x)) std::cout << iggg << ", ";\
    std::cout << "]\n"; } while(0)

int main(int argc, char **args) {
    int n = 1000, rep = 100, num_threads = 8, seed = 1, cs_seed = 2;
    bool timetest = false, debug = false;

    if (argc == 1) {
        std::cout << "Usage: " << args[0]
            << " n [--rep=REPEATS] [--num_threads=NUM] [--seed=SEED] [--cs_seed=CS_SEED] [--timetest] [--debug]\n\n"
            << "  --timetest  measure average time of mat-vec product, dot product and linear combination\n"
            << "    Without timetest will test operations once\n"
            << "  --debug  print matrices and vectors to stdout\n"
            << "  --seed=SEED  set seed for generating data\n"
            << "  --cs_seed=CS_SEED  set seed for control sums\n"
            << "  --rep=REPEATS  set number of repeats for averaging time\n"
            << "  --num_threads=NUM  set number of threads for OpenMP ";

        #ifndef _OPENMP
        std::cout << "(if compiled with it)";
        #endif


        std::cout << std::endl;

        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        std::string arg(args[i]);
        if (arg == "--timetest") {
            timetest = true;
        }
        else if (arg == "--debug") {
            debug = true;
        }
        else if (arg.find("--cs_seed=") == 0) {
            cs_seed = std::stoi(arg.substr(10));
        }
        else if (arg.find("--seed=") == 0) {
            seed = std::stoi(arg.substr(7));
        }
        else if (arg.find("--num_threads=") == 0) {
            num_threads = std::stoi(arg.substr(14));
        }
        else if (arg.find("--rep=") == 0) {
            rep = std::stoi(arg.substr(6));
        }
    }
    if (argc > 1) {
        n = std::stoi(args[1]);
    }

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    std::cout.precision(16);

    if (debug) {
        std::cout << "n=" << n << " rep=" << rep << " seed=" << seed << " cs_seed=" << cs_seed
            << " num_threads=" << num_threads << std::endl;
    }
    if (!timetest) {
        test_dot_product(n, seed, debug);
        test_mat_vec_product(n, seed, cs_seed, debug);
        test_linear_combination(n, seed, cs_seed, debug);
    }
    else {
        std::vector<double> csum;
        std::cout << "avg time of mat-vec product, n=" << n << ", repeats=" << rep
            << ": " << avg_time_of_mat_vec_product(n, rep, seed, cs_seed, csum) << std::endl;
        PRINT_VECTOR(csum, "  control sum");

        std::cout << "avg time of linear combination, n=" << n << ", repeats=" << rep
            << ": " << avg_time_of_linear_combination(n, rep, seed, cs_seed, csum) << std::endl;
        PRINT_VECTOR(csum, "  control sum");

        double dot = -1;
        std::cout << "avg time of dot product, n=" << n << ", repeats=" << rep
            << ": " << avg_time_of_dot_product(n, rep, seed, dot) << std::endl
            << "  dot : " << dot << std::endl;
    }
}
