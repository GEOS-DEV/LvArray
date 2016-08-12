#!/bin/bash

clang++        -std=c++14 -O3 -march=native   -I../src -o clang.x   arrayAccessorPerformance.cpp
clang++-mp-3.7 -std=c++14 -O3 -march=native   -I../src -o clang37.x arrayAccessorPerformance.cpp
g++-mp-5       -std=c++14 -O3 -march=westmere -I../src -o gcc5.x    arrayAccessorPerformance.cpp
g++-mp-4.9     -std=c++14 -O3 -march=westmere -I../src -o gcc49.x   arrayAccessorPerformance.cpp

clang++        -std=c++14 -O3 -march=native   -I../src -o clang.x2   arrayAccessorPerformance2.cpp
clang++-mp-3.7 -std=c++14 -O3 -march=native   -I../src -o clang37.x2 arrayAccessorPerformance2.cpp
g++-mp-5       -std=c++14 -O3 -march=westmere -I../src -o gcc5.x2    arrayAccessorPerformance2.cpp
g++-mp-4.9     -std=c++14 -O3 -march=westmere -I../src -o gcc49.x2   arrayAccessorPerformance2.cpp

echo clang-apple
for i in `seq 1 2`;
do
    ./clang.x   500 10 4000 100 2 2
done

echo clang-37
for i in `seq 1 2`;
do
    ./clang37.x 500 10 4000 100 2 2
done

echo gcc49
for i in `seq 1 2`;
do
    ./gcc49.x   500 10 4000 100 2 2
done

echo gcc5
for i in `seq 1 2`;
do
    ./gcc5.x    500 10 4000 100 2 2
done




echo clang-apple
for i in `seq 1 2`;
do
    ./clang.x2   500 10 4000 100 2 2
done

echo clang-37
for i in `seq 1 2`;
do
    ./clang37.x2 500 10 4000 100 2 2
done

echo gcc49
for i in `seq 1 2`;
do
    ./gcc49.x2   500 10 4000 100 2 2
done

echo gcc5
for i in `seq 1 2`;
do
    ./gcc5.x2    500 10 4000 100 2 2
done

