#!/bin/bash
. /usr/local/tools/dotkit/init.sh
use gcc-4.9.3p
use clang-3.8.1

#g++      -std=c++14 -O3 -I../src -o gcc49.x    arrayAccessorPerformance.cpp
#clang++  -std=c++14 -O3 -I../src -o clang38.x  arrayAccessorPerformance.cpp

#g++      -std=c++14 -O3 -I../src -o gcc49.x2    arrayAccessorPerformance2.cpp
#clang++  -std=c++14 -O3 -I../src -o clang38.x2  arrayAccessorPerformance2.cpp

echo gcc49
for i in `seq 1 10`;
do
    ./gcc49.x    $1 $2 $3 100 2 2
done

echo clang38
for i in `seq 1 10`;
do
    ./clang38.x  $1 $2 $3 100 2 2
done

echo gcc49-2
for i in `seq 1 10`;
do
    ./gcc49.x2   $1 $2 $3 100 2 2
done

echo clang38-2
for i in `seq 1 10`;
do
    ./clang38.x2 $1 $2 $3 100 2 2
done
