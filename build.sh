#!/usr/bin/env sh

mkdir build 2> /dev/null
g++ main.cpp matrices.cpp test.cpp timetest.cpp -Wall -Wextra -Wpedantic -fopenmp -O3 -Ofast -o build/main

echo
echo Compiled to build/main
