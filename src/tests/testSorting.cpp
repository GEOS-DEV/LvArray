/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "sorting.hpp"
#include "testUtils.hpp"
#include "Logger.hpp"
#include "Array.hpp"
#include "gtest/gtest.h"

#ifdef USE_CUDA

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include <chrono>
#include <random>
#include <algorithm>

using INDEX_TYPE = std::ptrdiff_t;

template <class T>
using Array1d = LvArray::Array<T, 1>;

template <class T>
using ArrayView1d = LvArray::ArrayView<T, 1> const;

namespace internal
{

// A static random number generator is used so that subsequent calls to the same test method will
// do different things. This initializes the generator with the default seed so the test behavior will
// be uniform across runs and platforms. However the behavior of an individual method will depend on the state
// of the generator and hence on the calls before it.
static std::mt19937_64 gen;

template <class T, class Compare>
void check(Array1d<T> & a, Array1d<T> & b, Compare comp)
{
  std::sort(a.begin(), a.end(), comp);
  sorting::makeSorted(b.begin(), b.end(), comp);

  for (INDEX_TYPE i = 0; i < a.size(); ++i)
  {
    EXPECT_EQ(a[i], b[i]);
  }
}

template <class T>
void fillArrays(INDEX_TYPE size, Array1d<T> & a, Array1d<T> & b)
{
  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, size);

  a.resize(size);
  b.resize(size);
  for (INDEX_TYPE i = 0; i < size; ++i)
  {
    T const val = T(valueDist(gen));
    a[i] = val;
    b[i] = val;
  }
}

template <class T, class Compare>
void correctnessTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 500;
  Array1d<T> a(MAX_SIZE);
  Array1d<T> b(MAX_SIZE);
  for (INDEX_TYPE size = 0; size < MAX_SIZE; ++size)
  {
    fillArrays<T>(size, a, b);
    check(a, b, Compare());
  }
}

template <class T, class Compare>
void performanceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 1024 * 1024;
  constexpr INDEX_TYPE NITER = 100;

  Array1d<T> a(MAX_SIZE);
  Array1d<T> b(MAX_SIZE);

  for (INDEX_TYPE size = 2; size <= MAX_SIZE; size *= 2)
  {
    std::chrono::duration<double> sortingTime;
    std::chrono::duration<double> stdTime;
    for (INDEX_TYPE iter = 0; iter < NITER; ++iter)
    {
      fillArrays<T>(size, a, b);

      auto start = std::chrono::high_resolution_clock::now();
      sorting::makeSorted(a.begin(), a.end(), Compare());
      auto end = std::chrono::high_resolution_clock::now();
      sortingTime += end - start;

      start = std::chrono::high_resolution_clock::now();
      std::sort(b.begin(), b.end(), Compare());
      end = std::chrono::high_resolution_clock::now();
      stdTime += end - start;
    }

    std::cout << "size = " << size << std::endl;
    std::cout << "    sorting::makeSorted : " << sortingTime.count() << std::endl;
    std::cout << "    std::sort     : " << stdTime.count() << std::endl << std::endl;
  }
}

#ifdef USE_CUDA

template <class T, class Compare>
void correctnessDeviceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 100;
  Array1d<T> a(MAX_SIZE);
  Array1d<T> b(MAX_SIZE);
  for (INDEX_TYPE size = 0; size < MAX_SIZE; ++size)
  {
    fillArrays<T>(size, a, b);

    ArrayView1d<T> & aView = a;
    forall(cuda(), 0, 1,
      [=] __device__ (INDEX_TYPE i)
      {
        sorting::makeSorted(aView.begin(), aView.end(), Compare());
      }
    );

    a.move(chai::CPU);

    std::sort(b.begin(), b.end(), Compare());

    for (INDEX_TYPE i = 0; i < a.size(); ++i)
    {
      EXPECT_EQ(a[i], b[i]);
    }
  }
}

#endif

} // namespace internal

TEST(sorting, correctness)
{
  internal::correctnessTest<int, sorting::less<int>>();
  internal::correctnessTest<int, sorting::greater<int>>();

  internal::correctnessTest<Tensor, sorting::less<Tensor>>();
  internal::correctnessTest<Tensor, sorting::greater<Tensor>>();

  internal::correctnessTest<TestString, sorting::less<TestString>>();
  internal::correctnessTest<TestString, sorting::greater<TestString>>();
}

TEST(sorting, performance)
{
  // internal::performanceTest<int, sorting::less<int>>();
  SUCCEED();
}

#ifdef USE_CUDA

CUDA_TEST(sorting, correctnessDevice)
{
  internal::correctnessDeviceTest<int, sorting::less<int>>();
  internal::correctnessDeviceTest<int, sorting::greater<int>>();

  internal::correctnessDeviceTest<Tensor, sorting::less<Tensor>>();
  internal::correctnessDeviceTest<Tensor, sorting::greater<Tensor>>();
}

#endif

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();
  MPI_Finalize();
  return result;
}