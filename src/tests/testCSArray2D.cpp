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

#include "CSArray2D.hpp"
#include "testUtils.hpp"
#include "gtest/gtest.h"
#include "Array.hpp"

#ifdef USE_CUDA

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include <vector>

namespace LvArray
{

using INDEX_TYPE = std::ptrdiff_t;

template <class T>
using ViewType = CSArray2DView<T, INDEX_TYPE const>;

namespace internal
{

/**
 * @brief Check that the given CSArray2DView is equivalent to the given vector of vectors.
 * @param [in] v the CSArray2DView to check.
 * @param [in] vRef the vector of vectors to check against.
 */
template <class T>
void compareToReference(ViewType<T const> const & v,
                        std::vector<std::vector<T>> const & vRef)
{
  ASSERT_EQ(v.size(), vRef.size());
  ASSERT_EQ(v.empty(), vRef.empty());
  if (v.empty())
  {
    ASSERT_EQ(v.size(), 0);
    return;
  }

  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    ASSERT_EQ(v.size(i), vRef[i].size());
    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      EXPECT_EQ(v(i, j), vRef[i][j]);
      EXPECT_EQ(&v(i, j), &v[i][j]);
    }
  }
}

/**
 * @brief Populate the given vector with values.
 * @param [out] v the vector to populate
 * @param [in] nVals the number of values to append.
 * @param [in] curIndex the value to begin at.
 */
template <class T >
void generateSubArray(std::vector<T> & v, INDEX_TYPE const nVals, INDEX_TYPE const curIndex)
{
  v.resize(nVals);
  for (INDEX_TYPE j = 0; j < nVals; ++j)
  {
    v[ j ] = T(curIndex + j);
  }
}

/**
 * @brief Test the appendArray method of CSArray2D.
 * @param [out] v the CSArray2D to test, must be empty.
 * @param [in] nArrays the number of arrays to append.
 * @param [in] maxNVals the maximum size of a sub array.
 * @return a vector of vectors serving as a reference.
 */
template <class T>
std::vector<std::vector<T>>
appendArrayTest(CSArray2D<T> & v, INDEX_TYPE const nArrays, INDEX_TYPE const maxNVals)
{
  EXPECT_TRUE(v.empty());

  std::vector<std::vector<T>> vRef;
  std::vector< T > vAppend(maxNVals);
  INDEX_TYPE curIndex = 0;
  INDEX_TYPE curNVals = 1;
  for (INDEX_TYPE i = 0; i < nArrays; ++i)
  {
    generateSubArray(vAppend, curNVals, curIndex);
    curIndex += curNVals;

    v.appendArray(vAppend.data(), curNVals);
    vRef.push_back(vAppend);

    curNVals = (curNVals + 1) % (maxNVals + 1);
  }

  compareToReference(v.toViewC(), vRef);
  return vRef;
}

/**
 * @brief Test the resizeArray method of the CSArray2D.
 * @param [in/out] v the CSArray2D to test.
 * @param [in/out] vRef the reference.
 */
template <class T>
void resizeArrayTest(CSArray2D<T>& v, std::vector<std::vector<T>> & vRef)
{
  INDEX_TYPE const nArrays = v.size();
  for (INDEX_TYPE i = 0; i < nArrays; ++i)
  {
    T const & defaultValue = T(i); 
    
    INDEX_TYPE newSize;
    if (i % 2 == 0) /* double the size. */
    {
      newSize = v.size(i) * 2; 
    }
    else /* halve the size. */
    {
      newSize = v.size(i) / 2; 
    }

    v.resizeArray(i, newSize, defaultValue);
    vRef[i].resize(newSize, defaultValue);
  }

  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the resizeNumArrays method of the CSArray2D.
 * @param [in/out] v the CSArray2D to test, must be empty.
 * @param [in] numArrays the number of arrays.
 * @param [in] maxNVals the maximum size an array.
 * @return a vector of vectors serving as a reference.
 */
template <class T>
std::vector<std::vector<T>>
resizeNumArraysTest(CSArray2D<T>& v, INDEX_TYPE const numArrays, INDEX_TYPE const maxNVals)
{
  EXPECT_TRUE(v.empty());

  v.resizeNumArrays(numArrays);

  std::vector<std::vector<T>> vRef(numArrays);
  std::vector<T> vAppend(maxNVals);
  INDEX_TYPE curIndex = 0;
  INDEX_TYPE curNVals = 1;
  for (INDEX_TYPE i = 0; i < numArrays; ++i)
  {
    generateSubArray(vAppend, curNVals, curIndex);
    curIndex += curNVals;

    v.resizeArray(i, curNVals);

    v.setArray(i, vAppend.data());
    vRef[i] = vAppend;

    curNVals = (curNVals + 1) % (maxNVals + 1);
  }

  compareToReference(v.toViewC(), vRef);
  return vRef;
}

/**
 * @brief Test the insertArray method of the CSArray2D.
 * @param [in/out] v the CSArray2D to test, must be empty.
 * @param [in] numArrays the number of arrays to insert.
 * @param [in] maxNVals the maximum size an array.
 * @return a vector of vectors serving as a reference.
 */
template <class T>
std::vector<std::vector<T>>
insertArrayTest(CSArray2D<T> & v, INDEX_TYPE const numArrays, INDEX_TYPE const maxNVals)
{
  EXPECT_TRUE(v.empty());

  std::vector<std::vector<T>> vRef;
  std::vector< T > vInsert(maxNVals);
  INDEX_TYPE curIndex = 0;
  INDEX_TYPE curNVals = 1;
  for (INDEX_TYPE i = 0; i < numArrays; ++i)
  {
    generateSubArray(vInsert, curNVals, curIndex);
    curIndex += curNVals;

    if( i % 3 == 0 )    // Insert at the beginning.
    {
      v.insertArray(0, vInsert.data(), curNVals);
      vRef.insert(vRef.begin(), vInsert);
    }
    else if( i % 3 == 1 )   // Insert at the end.
    {
      v.insertArray(v.size(), vInsert.data(), curNVals);
      vRef.insert(vRef.end(), vInsert);
    }
    else  // Insert in the middle.
    {
      v.insertArray(v.size() / 2, vInsert.data(), curNVals);
      vRef.insert(vRef.begin() + vRef.size() / 2, vInsert);
    }

    curNVals = (curNVals + 1) % (maxNVals + 1);
  }

  compareToReference(v.toViewC(), vRef);
  return vRef;
}

/**
 * @brief Test the removeArray method of the CSArray2D.
 * @param [in/out] v the CSArray2D to test.
 * @param [in/out] vRef the reference.
 */
template <class T>
void removeArrayTest(CSArray2D<T> & v, std::vector<std::vector<T>> & vRef)
{
  const INDEX_TYPE n_elems = v.size();
  for (INDEX_TYPE i = 0; i < n_elems; ++i)
  {
    if (i % 3 == 0)    // erase the beginning.
    {
      v.removeArray(0);
      vRef.erase(vRef.begin());
    }
    else if (i % 3 == 1)   // erase at the end.
    {
      v.removeArray(v.size() - 1);
      vRef.erase(vRef.end() - 1);
    }
    else  // erase the middle.
    {
      v.removeArray(v.size() / 2);
      vRef.erase(vRef.begin() + vRef.size() / 2);
    }

    if (i % 10 == 0)
    {
      compareToReference(v.toViewC(), vRef);
    }
  }

  ASSERT_TRUE(v.empty());
  compareToReference(v.toViewC(), vRef);
}

/**
 * @brief Test the CSArray2D copy constructor.
 * @param [in] v the CSArray2D to test.
 */
template <class T>
void deepCopyTest(CSArray2D<T> const & v)
{
  CSArray2D<T> vCpy(v);

  ASSERT_EQ(v.size(), vCpy.size());
  ASSERT_NE(v.getOffsets(), vCpy.getOffsets());
  ASSERT_NE(v.getValues(), vCpy.getValues());

  INDEX_TYPE curIndex = 0;  
  for( INDEX_TYPE i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ(v.size(i), vCpy.size(i));

    for (INDEX_TYPE j = 0 ; j < v.size(i) ; ++j)
    {
      EXPECT_EQ(v[i][j], vCpy[i][j]);
      EXPECT_EQ(v[i][j], T(curIndex));
      vCpy[i][j] = T(2 * curIndex);
      ++curIndex;
    }
  }

  curIndex = 0;
  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    ASSERT_EQ(v.size(i), vCpy.size(i));

    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      EXPECT_EQ(v[i][j], T(curIndex));
      EXPECT_EQ(vCpy[i][j], T(2 * curIndex));
      ++curIndex;
    }
  }
}

/**
 * @brief Test the CSArray2DView copy constructor.
 * @param [in] v the CSArray2D to test.
 */
template <class T>
void shallowCopyTest(CSArray2D<T> const & v)
{
  ViewType<T> vCpy(v);

  ASSERT_EQ(v.size(), vCpy.size());
  ASSERT_EQ(v.getOffsets(), vCpy.getOffsets());
  ASSERT_EQ(v.getValues(), vCpy.getValues());

  INDEX_TYPE curIndex = 0;
  for( INDEX_TYPE i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ(v.size(i), vCpy.size(i));

    for (INDEX_TYPE j = 0 ; j < v.size(i) ; ++j)
    {
      EXPECT_EQ(v[i][j], vCpy[i][j]);
      EXPECT_EQ(v[i][j], T(curIndex));
      vCpy[i][j] = T(2 * curIndex);
      ++curIndex;
    }
  }

  curIndex = 0;
  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    ASSERT_EQ(v.size(i), vCpy.size(i));

    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      EXPECT_EQ(v[i][j], vCpy[i][j]);
      EXPECT_EQ(v[i][j], T(2 * curIndex));
      ++curIndex;
    }
  }
}

#ifdef USE_CUDA

/**
 * @brief Test the CSArray2DView copy constructor in regards to memory motion.
 * @param [in/out] v the CSArray2DView to test.
 */
template <class T>
void memoryMotionTest(ViewType<T> const & v)
{
  INDEX_TYPE const N = v.size();

  // Set the values.
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE i = 0; i < N; ++i)
  {
    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      v(i, j) = T(curIndex++);
    }
  }

  // Capture the view and change the values on device.
  forall(cuda(), 0, N,
    [=] __device__ (INDEX_TYPE i)
    {
      for (INDEX_TYPE j = 0; j < v.size(i); ++j)
      {
        v(i, j) += v(i, j);
        v[i][j] += v[i][j];
      }
    }
  );

  // Capture the view and check that the values have been updated.
  curIndex = 0;
  forall(sequential(), 0, N,
    [=, &curIndex](INDEX_TYPE i)
    {
      for (INDEX_TYPE j = 0; j < v.size(i); ++j)
      {
        T val = T(curIndex++);
        val += val;
        val += val;
        EXPECT_EQ(val, v(i, j));
      }
    }
  );
}

/**
 * @brief Test the CSArray2D move method.
 * @param [in/out] v the CSArray2D to test.
 */
template <class T>
void memoryMotionMoveTest(CSArray2D<T> & v)
{
  INDEX_TYPE const N = v.size();

  // Upcast to a CSArray2DView and set the values.
  ViewType<T> const & view = v;
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE i = 0; i < view.size(); ++i)
  {
    for (INDEX_TYPE j = 0; j < view.size(i); ++j)
    {
      view(i, j) = T(curIndex++);
    }
  }

  // Capture the view and change the values on device.
  forall(cuda(), 0, N,
    [=] __device__ (INDEX_TYPE i)
    {
      for (INDEX_TYPE j = 0; j < view.size(i); ++j)
      {
        view(i, j) += view(i, j);
        view[i][j] += view[i][j];
      }
    }
  );

  // Move the CSArray2D and check that the values have been updated.
  v.move(chai::CPU);
  curIndex = 0;
  for (INDEX_TYPE i = 0; i < view.size(); ++i)
  {
    for (INDEX_TYPE j = 0; j < view.size(i); ++j)
    {
      T val = T(curIndex++);
      val += val;
      val += val;
      EXPECT_EQ(val, view(i, j));
    }
  }
}

/**
 * @brief Test the CSArray2DView<T const> semantics.
 * @param [in/out] v the CSArray2DView to test.
 */
template <class T>
void memoryMotionConstTest(ViewType<T> const & v)
{
  INDEX_TYPE const N = v.size();

  // Set the values.
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      v(i, j) = T(curIndex++);
    }
  }

  // Create a const view and capture it on device.
  ViewType<T const> const & vConst = v; 
  forall(cuda(), 0, N,
    [=] __device__ (INDEX_TYPE i)
    {
      INDEX_TYPE const offset = vConst.getOffsets()[i];
      for (INDEX_TYPE j = 0; j < vConst.size(i); ++j)
      {
        GEOS_ASSERT(vConst(i, j) == T(offset + j));
      }
    }
  );

  // Change the values of the original view.
  curIndex = 0;
  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      v(i, j) = T(2 * curIndex++);
    }
  }

  // Capture the view on host and check that the values haven't been overwritten.
  curIndex = 0;
  forall(sequential(), 0, N,
    [=, &curIndex](INDEX_TYPE i)
    {
      for (INDEX_TYPE j = 0; j < v.size(i); ++j)
      {
        EXPECT_EQ(v(i, j), T(2 * curIndex++));
      }
    }
  );
}

/**
 * @brief Test the CSArray2DView setArray method on the device.
 * @param [in/out] v the CSArray2DView to test.
 */
template <class T>
void setArrayDeviceTest(ViewType<T> const & v)
{
  INDEX_TYPE const N = v.size();

  INDEX_TYPE totalSize = 0;
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE i = 0; i < v.size(); ++i)
  {
    totalSize += v.size(i);
    for (INDEX_TYPE j = 0; j < v.size(i); ++j)
    {
      v(i, j) = T(curIndex++);
    }
  }

  Array<T, 1> arr(totalSize);
  ArrayView<T, 1> const & arrView = arr;

  forall(cuda(), 0, N,
    [=] __device__ (INDEX_TYPE i)
    {
      INDEX_TYPE const offset = v.getOffsets()[i];
      for (INDEX_TYPE j = 0; j < v.size(i); ++j)
      {
        INDEX_TYPE const curIndex = offset + j;
        arrView[curIndex] = T(2 * curIndex);
      }

      v.setArray(i, &arrView[offset]);
    }
  );

  curIndex = 0;
  forall(sequential(), 0, N,
    [=, &curIndex](INDEX_TYPE i)
    {
      for (INDEX_TYPE j = 0; j < v.size(i); ++j)
      {
        EXPECT_EQ(v(i, j), T(2 * curIndex++));
      }
    }
  );
}

#endif

} /* namespace internal */

TEST(CSArray2D, appendArray)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
  }

  {
    CSArray2D<TestString> v;
    internal::appendArrayTest(v, N, M);
  }
}

TEST(CSArray2D, resizeArray)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    std::vector<std::vector<int>> vRef = internal::appendArrayTest(v, N, M);
    internal::resizeArrayTest(v, vRef);
  }

  {
    CSArray2D<Tensor> v;
    std::vector<std::vector<Tensor>> vRef = internal::appendArrayTest(v, N, M);
    internal::resizeArrayTest(v, vRef);
  }

  {
    CSArray2D<TestString> v;
    std::vector<std::vector<TestString>> vRef = internal::appendArrayTest(v, N, M);
    internal::resizeArrayTest(v, vRef);
  }
}

TEST(CSArray2D, resizeNumArrays)
{
  constexpr INDEX_TYPE N = 100;      /* Number of arrays to create */
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::resizeNumArraysTest(v, N, M);
  }

  {
    CSArray2D<Tensor> v;
    internal::resizeNumArraysTest(v, N, M);
  }

  {
    CSArray2D<TestString> v;
    internal::resizeNumArraysTest(v, N, M);
  }
}

TEST(CSArray2D, insertArray)
{
  constexpr INDEX_TYPE N = 100;       /* Number of times to call insert */
  constexpr INDEX_TYPE M = 10;        // Max size of each array.

  {
    CSArray2D<int> v;
    internal::insertArrayTest(v, N, M);
  }

  {
    CSArray2D<Tensor> v;
    internal::insertArrayTest(v, N, M);
  }

  {
    CSArray2D<TestString> v;
    internal::insertArrayTest(v, N, M);
  }
}

TEST(CSArray2D, removeArray)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    std::vector<std::vector<int>> vRef = internal::appendArrayTest(v, N, M);
    internal::removeArrayTest(v, vRef);
  }

  {
    CSArray2D<Tensor> v;
    std::vector<std::vector<Tensor>> vRef = internal::appendArrayTest(v, N, M);
    internal::removeArrayTest(v, vRef);
  }

  {
    CSArray2D<TestString> v;
    std::vector<std::vector<TestString>> vRef = internal::appendArrayTest(v, N, M);
    internal::removeArrayTest(v, vRef);
  }
}

TEST(CSArray2D, deepCopy)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::deepCopyTest(v);
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::deepCopyTest(v);
  }

  {
    CSArray2D<TestString> v;
    internal::appendArrayTest(v, N, M);
    internal::deepCopyTest(v);
  }
}

TEST(CSArray2D, shallowCopy)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::shallowCopyTest(v);
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::shallowCopyTest(v);
  }

  {
    CSArray2D<TestString> v;
    internal::appendArrayTest(v, N, M);
    internal::shallowCopyTest(v);
  }
}

#ifdef USE_CUDA

CUDA_TEST(CSArray2D, memoryMotion)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionTest(v.toView());
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionTest(v.toView());
  }
}

CUDA_TEST(CSArray2D, memoryMotionMove)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionMoveTest(v);
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionMoveTest(v);
  }
}

CUDA_TEST(CSArray2D, memoryMotionConst)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionConstTest(v.toView());
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::memoryMotionConstTest(v.toView());
  }
}

CUDA_TEST(CSArray2D, setArrayDevice)
{
  constexpr INDEX_TYPE N = 100;      // Number of arrays to append.
  constexpr INDEX_TYPE M = 10;       // Max size of each array.

  {
    CSArray2D<int> v;
    internal::appendArrayTest(v, N, M);
    internal::setArrayDeviceTest(v.toView());
  }

  {
    CSArray2D<Tensor> v;
    internal::appendArrayTest(v, N, M);
    internal::setArrayDeviceTest(v.toView());
  }
}

#endif

} // namespace LvArray

int main( int argc, char* argv[] )
{
  logger::InitializeLogger();

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();

#ifdef USE_CHAI
  chai::ArrayManager::finalize();
#endif

  return result;
}
