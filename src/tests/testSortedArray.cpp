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

#include "SortedArray.hpp"
#include "testUtils.hpp"
#include "gtest/gtest.h"

#ifdef USE_CUDA

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include <vector>
#include <set>
#include <random>

namespace LvArray
{

using INDEX_TYPE = std::ptrdiff_t;

namespace internal
{

// A static random number generator is used so that subsequent calls to the same test method will
// do different things. This initializes the generator with the default seed so the test behavior will
// be uniform across runs and platforms. However the behavior of an individual method will depend on the state
// of the generator and hence on the calls before it.
static std::mt19937_64 gen;

/**
 * @brief Check that the SortedArrayView is equivalent to the std::set. Checks equality using the
 * operator[] and the raw pointer.
 * @param [in] v the SortedArrayView to check.
 * @param [in] vRef the std::set to check against.
 */
template <class T>
void compareToReference(SortedArrayView<T const> const & v, const std::set<T> & vRef )
{
  ASSERT_EQ( v.size(), vRef.size() );
  ASSERT_EQ( v.empty(), vRef.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  T const * ptr = v.values();
  typename std::set< T >::const_iterator it = vRef.begin();
  for(int i = 0 ; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], *it );
    ASSERT_EQ( ptr[ i ], *it );
    ++it;
  }
}

/**
 * @brief Test the insert method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_INSERTS the number of times to call insert.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void insertTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_INSERTS,
                 INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
  {
    T const value = T(valueDist(gen));
    ASSERT_EQ(vRef.insert(value).second, v.insert(value));
  }

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the insert multiple sorted method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_INSERTS the number of values to insert at a time.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void insertMultipleSortedTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_INSERTS,
                              INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::vector<T> values(MAX_INSERTS);

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
  {
    values[i] = T(valueDist(gen));
  }

  std::sort(values.begin(), values.end());
  v.insertSorted(values.data(), MAX_INSERTS);
  vRef.insert(values.begin(), values.end());

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the insert multiple method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_INSERTS the number of values to insert at a time.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void insertMultipleTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_INSERTS,
                        INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::vector<T> values(MAX_INSERTS);

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
  {
    values[i] = T(valueDist(gen));
  }

  v.insert(values.data(), MAX_INSERTS);
  vRef.insert(values.begin(), values.end());

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the erase method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_REMOVES the number of times to call erase.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void eraseTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_REMOVES,
                 INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
  {
    T const value = T(valueDist(gen));
    ASSERT_EQ(vRef.erase(value), v.erase(value));
  }

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the erase multiple sorted method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_INSERTS the number of values to erase at a time.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void eraseMultipleSortedTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_REMOVES,
                          INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::vector<T> values(MAX_REMOVES);

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
  {
    values[i] = T(valueDist(gen));
  }

  std::sort(values.begin(), values.end());
  v.eraseSorted(values.data(), MAX_REMOVES);

  for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
  {
    vRef.erase(values[i]);
  }

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the erase multiple method of the SortedArray.
 * @param [in/out] v the SortedArray to test.
 * @param [in/out] vRef the std::set to check against.
 * @param [in] MAX_INSERTS the number of values to erase at a time.
 * @param [in] MAX_VAL the largest value possibly generate.
 */
template <class T>
void eraseMultipleTest(SortedArray<T> & v, std::set<T> & vRef, INDEX_TYPE const MAX_REMOVES,
                       INDEX_TYPE const MAX_VAL)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  std::vector<T> values(MAX_REMOVES);

  std::uniform_int_distribution<INDEX_TYPE> valueDist(0, MAX_VAL);

  for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
  {
    values[i] = T(valueDist(gen));
  }

  v.erase(values.data(), MAX_REMOVES);

  for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
  {
    vRef.erase(values[i]);
  }

  compareToReference(v.toView(), vRef);
}

/**
 * @brief Test the contains method of the SortedArray.
 * @param [in] v the SortedArrayView to check.
 * @param [in] vRef the std::set to check against.
 */
template <class T>
void containsTest(SortedArrayView<T const> const & v, std::set<T> & vRef)
{
  INDEX_TYPE const N = v.size();
  ASSERT_EQ(N, vRef.size());

  compareToReference(v, vRef);

  typename std::set<T>::iterator it = vRef.begin();
  for (INDEX_TYPE i = 0; i < N; ++i)
  {
    T const & value = *it;
    EXPECT_TRUE(v.contains(value));
    EXPECT_EQ(v.contains(T(i)), vRef.count(T(i)));
    ++it;
  }
}

/**
 * @brief Test the copy constructor of the SortedArray.
 * @param [in] v the SortedArrayView to check.
 * @param [in] vRef the std::set to check against.
 */
template <class T>
void deepCopyTest(SortedArray<T> const & v, std::set<T> const & vRef)
{
  INDEX_TYPE const N = v.size();

  SortedArray<T> v_cpy(v);
  ASSERT_EQ(N, v_cpy.size());

  T const * const vals = v.values();
  T const * const vals_cpy = v_cpy.values();
  ASSERT_NE(vals, vals_cpy);

  // Iterate backwards and erase entries from v_cpy.
  for (INDEX_TYPE i = N - 1; i >= 0; --i)
  {
    EXPECT_EQ(vals[i], vals_cpy[i]);
    v_cpy.erase(vals_cpy[i]);
  }

  EXPECT_EQ(v_cpy.size(), 0);
  EXPECT_EQ(v.size(), N);

  // check that v is still the same as vRef.
  compareToReference(v.toView(), vRef);
}

#ifdef USE_CUDA

/**
 * @brief Test the SortedArrayView copy constructor in regards to memory motion.
 * @param [in/out] v the SortedArray to test, must be empty.
 * @param [in] SIZE the number of values to insert.
 */
template <class T>
void memoryMotionTest(SortedArray<T> & v, INDEX_TYPE const SIZE)
{
  ASSERT_TRUE(v.empty());

  // Insert values.
  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    v.insert(T(i));
  }

  ASSERT_EQ(v.size(), SIZE);

  // Capture a view on the device.
  SortedArrayView<T const> const & vView = v;
  forall(cuda(), 0, SIZE,
    [=] __device__ (INDEX_TYPE i)
    {
      GEOS_ERROR_IF(vView[i] != T(i), "Values changed when moved.");
    }
  );

  // Change the values.
  v.clear();
  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    v.insert(T(i * i));
  }

  ASSERT_EQ(v.size(), SIZE);

  // Capture the view on host and check that the values haven't been overwritten.
  forall(sequential(), 0, SIZE,
    [=](INDEX_TYPE i)
    {
      EXPECT_EQ(v[i], T(i * i));
    }
  );
}

/**
 * @brief Test the SortedArray move method.
 * @param [in/out] v the SortedArray to test, must be empty.
 * @param [in] SIZE the number of values to insert.
 */
template <class T>
void memoryMotionMoveTest(SortedArray<T> & v, INDEX_TYPE SIZE)
{
  ASSERT_TRUE(v.empty());

  // Insert values.
  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    v.insert(T(i));
  }

  ASSERT_EQ(v.size(), SIZE);

  // Capture a view on the device.
  SortedArrayView<T const> const & vView = v;
  forall(cuda(), 0, SIZE,
    [=] __device__ (INDEX_TYPE i)
    {
      GEOS_ERROR_IF(vView[i] != T(i), "Values changed when moved.");
    }
  );

  // Change the values.
  v.clear();
  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    v.insert(T(i * i));
  }

  ASSERT_EQ(v.size(), SIZE);

  // Move the array back to the host and check that the values haven't been overwritten.
  v.move(chai::CPU);
  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    EXPECT_EQ(v[i], T(i * i));
  }
}

/**
 * @brief Test the contains method of the SortedArray on device.
 * @param [in] v the SortedArrayView to check.
 * @param [in] SIZE the number of values to insert.
 */
template <class T>
void containsDeviceTest(SortedArray<T> & v, INDEX_TYPE SIZE)
{
  ASSERT_TRUE(v.empty());

  for (INDEX_TYPE i = 0; i < SIZE; ++i)
  {
    v.insert(T(2 * i));
  }

  ASSERT_EQ(v.size(), SIZE);

  SortedArrayView<T const> const & vView = v;
  forall(cuda(), 0, SIZE,
    [=] __device__ (INDEX_TYPE i)
    {
      GEOS_ERROR_IF(!vView.contains(T(2 * i)), "vView should contain even numbers.");
      GEOS_ERROR_IF(vView.contains(T(2 * i + 1)), "vView should not contain odd numbers.");
    }
  );
}

#endif

} /* namespace internal */


TEST(SortedArray, insert)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
  }
}

TEST(SortedArray, insertMultipleSorted)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleSortedTest(v, ref, MAX_INSERTS, MAX_VAL);
  }
}

TEST(SortedArray, insertMultiple)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::insertMultipleTest(v, ref, MAX_INSERTS, MAX_VAL);
  }
}

TEST(SortedArray, erase)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;
  constexpr INDEX_TYPE MAX_REMOVES = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseTest(v, ref, MAX_REMOVES, MAX_VAL);
  }
}

TEST(SortedArray, eraseMultipleSorted)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;
  constexpr INDEX_TYPE MAX_REMOVES = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleSortedTest(v, ref, MAX_REMOVES, MAX_VAL);
  }
}

TEST(SortedArray, eraseMultiple)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;
  constexpr INDEX_TYPE MAX_REMOVES = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::eraseMultipleTest(v, ref, MAX_REMOVES, MAX_VAL);
  }
}

TEST(SortedArray, contains)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::containsTest(v.toView(), ref);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::containsTest(v.toView(), ref);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::containsTest(v.toView(), ref);
  }
}

TEST(SortedArray, deepCopy)
{
  constexpr INDEX_TYPE MAX_VAL = 1000;
  constexpr INDEX_TYPE MAX_INSERTS = 200;

  {
    SortedArray<int> v;
    std::set<int> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::deepCopyTest(v, ref);
  }

  {
    SortedArray<Tensor> v;
    std::set<Tensor> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::deepCopyTest(v, ref);
  }

  {
    SortedArray<TestString> v;
    std::set<TestString> ref;
    internal::insertTest(v, ref, MAX_INSERTS, MAX_VAL);
    internal::deepCopyTest(v, ref);
  }
}

#ifdef USE_CUDA

TEST(SortedArray, memoryMotion)
{
  constexpr INDEX_TYPE SIZE = 1000;

  {
    SortedArray<int> v;
    internal::memoryMotionTest(v, SIZE);
  }

  {
    SortedArray<Tensor> v;
    internal::memoryMotionTest(v, SIZE);
  }
}

TEST(SortedArray, memoryMotionMove)
{
  constexpr INDEX_TYPE SIZE = 1000;

  {
    SortedArray<int> v;
    internal::memoryMotionMoveTest(v, SIZE);
  }

  {
    SortedArray<Tensor> v;
    internal::memoryMotionMoveTest(v, SIZE);
  }
}

TEST(SortedArray, containsDevice)
{
  constexpr INDEX_TYPE SIZE = 1000;

  {
    SortedArray<int> v;
    internal::containsDeviceTest(v, SIZE);
  }

  {
    SortedArray<Tensor> v;
    internal::containsDeviceTest(v, SIZE);
  }
}

#endif

} /* namespace LvArray */

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
