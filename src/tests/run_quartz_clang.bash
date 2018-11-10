#!/bin/bash

module add clang/4.0.0

rm *.o clang40.x
clang++-4.0.0 -std=c++14 -O3     -march=native -Wno-vla  -I../src -I. -c arrayAccessorPerformance.cpp
clang++-4.0.0 -std=c++14 -O3     -march=native -Wno-vla  -I../src -I. -c arrayAccessorPerformanceMain.cpp
clang++-4.0.0 -std=c++14 -O3     -march=native -o clang40.x  arrayAccessorPerformanceMain.o arrayAccessorPerformance.o

sleep 1
for i in `seq 1 10`;
do
  ./clang40.x    $1 $2 $3 100 2 1
    sleep 1
done