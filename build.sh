#!/usr/bin/env sh

mkdir build 2>/dev/null
# removed -Ofast because std::isfinite(nan) = 1
mpic++ *.cpp -fopenmp -std=c++17 -Wall -Wextra -Wpedantic -O3 -o build/mpi_main

echo
echo Compiled to build/mpi_main
