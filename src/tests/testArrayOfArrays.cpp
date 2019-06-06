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

#include "gtest/gtest.h"

#include "ArrayOfArrays.hpp"
#include "testUtils.hpp"

#ifdef USE_CUDA

#include "Array.hpp"

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include <vector>
#include <random>

namespace LvArray
{

using INDEX_TYPE = ::std::ptrdiff_t;

template <class T, bool CONST_SIZES=false>
using ViewType = ArrayOfArraysView<T, INDEX_TYPE const, CONST_SIZES>;

template <class T>
using REF_TYPE = std::vector<std::vector<T>>;

template <class T>
void compareToReference(ViewType<T const, true> const & view, REF_TYPE<T> const & ref)
{
  ASSERT_EQ(view.size(), ref.size());
  
  for (INDEX_TYPE i = 0; i < view.size(); ++i)
  {
    ASSERT_EQ(view.sizeOfArray(i), ref[i].size());

    for (INDEX_TYPE j = 0; j < view.sizeOfArray(i); ++j)
    {
      EXPECT_EQ(view[i][j], ref[i][j]);
      EXPECT_EQ(view(i, j), view[i][j]);
    }
  }
}

#define COMPARE_TO_REFERENCE(view, ref) { SCOPED_TRACE(""); compareToReference(view, ref); }

template <class T>
class ArrayOfArraysTest : public ::testing::Test
{
public:

  void appendArray( ArrayOfArrays<T> & array, REF_TYPE<T> & ref,
                     INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize)
  {
    compareToReference(array.toViewCC(), ref);

    std::vector<T> arrayToAppend(maxAppendSize);

    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxAppendSize);
      arrayToAppend.resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        arrayToAppend[j] = T(i * LARGE_NUMBER + j);
      }

      array.appendArray(arrayToAppend.data(), arrayToAppend.size());
      ref.push_back(arrayToAppend);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void insertArray( ArrayOfArrays<T> & array, REF_TYPE<T> & ref,
                    INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize)
  {
    compareToReference(array.toViewCC(), ref);

    std::vector<T> arrayToAppend(maxAppendSize);

    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxAppendSize);
      arrayToAppend.resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        arrayToAppend[j] = T(i * LARGE_NUMBER + j);
      }

      INDEX_TYPE insertPos = rand(0, array.size());
      array.insertArray(insertPos, arrayToAppend.data(), arrayToAppend.size());
      ref.insert(ref.begin() + insertPos, arrayToAppend);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void eraseArray( ArrayOfArrays<T> & array, REF_TYPE<T> & ref,
                    INDEX_TYPE const nArrays )
  {
    compareToReference(array.toViewCC(), ref);

    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const removePos = rand(0, array.size() -1);
      array.eraseArray(removePos);
      ref.erase(ref.begin() + removePos);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void resize(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const newSize = rand(0, 2 * array.size());
    array.resize(newSize);
    ref.resize(newSize);

    compareToReference(array.toViewCC(), ref);
  }

  void resizeArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxSize)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const newSize = rand(0, maxSize);
      
      T const defaultValue = T(i);
      array.resizeArray(i, newSize, defaultValue);
      ref[i].resize(newSize, defaultValue);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void appendToArray( ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxAppendSize)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxAppendSize);
      INDEX_TYPE const arraySize = array.sizeOfArray(i);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);
        array.appendToArray(i, value);
        ref[i].push_back(value);
      }
    }

    compareToReference(array.toViewCC(), ref);
  }

  void appendMultipleToArray( ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxAppendSize)
  {
    compareToReference(array.toViewCC(), ref);

    std::vector<T> valuesToAppend(maxAppendSize);
    INDEX_TYPE const nArrays = array.size();

    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxAppendSize);
      valuesToAppend.resize(nValues);
      INDEX_TYPE const arraySize = array.sizeOfArray(i);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        valuesToAppend[j] = T(i * LARGE_NUMBER + arraySize + j);
      }

      array.appendToArray(i, valuesToAppend.data(), valuesToAppend.size());
      ref[i].insert(ref[i].end(), valuesToAppend.begin(), valuesToAppend.end());
    }

    compareToReference(array.toViewCC(), ref);
  }

  void insertIntoArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxInsertSize)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const nValues = rand(0, maxInsertSize);
      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);

        INDEX_TYPE const insertPos = rand(0, arraySize + j);
        array.insertIntoArray(i, insertPos, value);
        ref[i].insert(ref[i].begin() + insertPos, value);
      }
    }

    compareToReference(array.toViewCC(), ref);
  }

  void insertMultipleIntoArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxInsertSize)
  {
    compareToReference(array.toViewCC(), ref);

    std::vector<T> valuesToAppend(maxInsertSize);
    
    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInsertSize);
      valuesToAppend.resize(nValues);
      
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        valuesToAppend[j] = T(i * LARGE_NUMBER + arraySize + j);
      }

      INDEX_TYPE const insertPos = rand(0, arraySize);

      array.insertIntoArray(i, insertPos, valuesToAppend.data(), valuesToAppend.size());
      ref[i].insert(ref[i].begin() + insertPos, valuesToAppend.begin(), valuesToAppend.end());
    }

    compareToReference(array.toViewCC(), ref);
  }

  void eraseFromArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxRemoves)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const nToRemove = rand(0, min(arraySize, maxRemoves));
      for (INDEX_TYPE j = 0; j < nToRemove; ++j)
      {
        INDEX_TYPE const removePosition = rand(0, arraySize - j - 1);
        
        array.eraseFromArray(i, removePosition);
        ref[i].erase(ref[i].begin() + removePosition);
      }
    }

    compareToReference(array.toViewCC(), ref);
  }

  void removeMultipleFromArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxRemoves)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      if (arraySize == 0) continue;

      INDEX_TYPE const removePosition = rand(0, arraySize - 1);
      INDEX_TYPE const nToRemove = rand(0, min(maxRemoves, arraySize - removePosition));
      
      array.eraseFromArray(i, removePosition, nToRemove);
      ref[i].erase(ref[i].begin() + removePosition, ref[i].begin() + removePosition + nToRemove);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void compress(ArrayOfArrays<T> & array, REF_TYPE<T> const & ref)
  {
    compareToReference(array.toViewCC(), ref);

    array.compress();
    T const * const values = array[0];

    INDEX_TYPE curOffset = 0;
    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      ASSERT_EQ(array.sizeOfArray(i), array.capacityOfArray(i));

      T const * const curValues = array[i];
      ASSERT_EQ(values + curOffset, curValues);

      curOffset += array.sizeOfArray(i);
    }

    compareToReference(array.toViewCC(), ref);
  }

  void fill(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    compareToReference(array.toViewCC(), ref);

    T const * const values = array[0];
    T const * const endValues = array[array.size() - 1];

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      fillArray(array, ref, i);
    }

    T const * const newValues = array[0];
    T const * const newEndValues = array[array.size() - 1];
    ASSERT_EQ(values, newValues);
    ASSERT_EQ(endValues, newEndValues);

    compareToReference(array.toViewCC(), ref);
  }

  void capacityOfArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const MAX_CAPACITY)
  {
    compareToReference(array.toViewCC(), ref);

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      INDEX_TYPE const capacity = array.capacityOfArray(i);
      INDEX_TYPE const newCapacity = rand(0, MAX_CAPACITY);
      
      if (newCapacity < capacity)
      {
        T const * const values = array[i];
        array.setCapacityOfArray(i, newCapacity);
        T const * const newValues = array[i];

        ASSERT_EQ(values, newValues);
        ref[i].resize(newCapacity);
      }
      else
      {
        array.setCapacityOfArray(i, newCapacity);
        fillArray(array, ref, i);
      }
    }

    compareToReference(array.toViewCC(), ref);
  }

  void deepCopy(ArrayOfArrays<T> const & array, REF_TYPE<T> const & ref)
  {
    compareToReference(array.toViewCC(), ref);

    ArrayOfArrays<T> copy(array);

    ASSERT_EQ(array.size(), copy.size());

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      ASSERT_EQ(arraySize, copy.sizeOfArray(i));
      
      T const * const values = array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_NE(values, valuesCopy);

      for (INDEX_TYPE j = 0; j < arraySize; ++j)
      {
        ASSERT_EQ(array[i][j], copy[i][j]);
        copy[i][j] = T(-i * LARGE_NUMBER - j);
      }
    }

    compareToReference(array.toViewCC(), ref);
  }

  void shallowCopy(ArrayOfArrays<T> const & array)
  {
    ViewType<T> const copy(array.toView());

    ASSERT_EQ(array.size(), copy.size());

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      ASSERT_EQ(arraySize, copy.sizeOfArray(i));
      T const * const values = array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_EQ(values, valuesCopy);

      for (INDEX_TYPE j = 0; j < arraySize; ++j)
      {
        ASSERT_EQ(array[i][j], copy[i][j]);

        copy[i][j] = T(-i * LARGE_NUMBER - j);

        ASSERT_EQ(array[i][j], T(-i * LARGE_NUMBER - j));
        ASSERT_EQ(array[i][j], copy[i][j]);

      }
    }
  }

protected:

  void fillArray(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const i)
  {
    INDEX_TYPE const arraySize = array.sizeOfArray(i);
    INDEX_TYPE const nToAppend = array.capacityOfArray(i) - arraySize;
    T const * const values = array[i];
    T const * const endValues = array[array.size() - 1];
    INDEX_TYPE const capacity = array.capacityOfArray(i);

    for (INDEX_TYPE j = 0; j < nToAppend; ++j)
    {
      T const valueToAppend(LARGE_NUMBER * i + j);
      array.appendToArray(i, valueToAppend);
      ref[i].push_back(valueToAppend);
    }

    T const * const newValues = array[i];
    T const * const newEndValues = array[array.size() - 1];
    INDEX_TYPE const endCapacity = array.capacityOfArray(i);
    ASSERT_EQ(values, newValues);
    ASSERT_EQ(endValues, newEndValues);
    ASSERT_EQ(capacity, endCapacity);
  }

  INDEX_TYPE rand(INDEX_TYPE const min, INDEX_TYPE const max)
  {
    return std::uniform_int_distribution<INDEX_TYPE>(min, max)(m_gen);
  }

  static constexpr INDEX_TYPE LARGE_NUMBER = 1E6;
  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types<INDEX_TYPE, Tensor, TestString>;
TYPED_TEST_CASE(ArrayOfArraysTest, TestTypes);

TYPED_TEST(ArrayOfArraysTest, construction)
{
  // Test empty construction
  {
    ArrayOfArrays<TypeParam> a;
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(a.capacity(), 0);
  }

  // Test construction with number of arrays
  {
    ArrayOfArrays<TypeParam> a(10);
    ASSERT_EQ(a.size(), 10);
    ASSERT_EQ(a.capacity(), 10);

    for (INDEX_TYPE i = 0; i < a.size(); ++i)
    {
      ASSERT_EQ(a.sizeOfArray(i), 0);
      ASSERT_EQ(a.capacityOfArray(i), 0);
    }
  }

  // Test construction with a number of arrays and a hint
  {
    ArrayOfArrays<TypeParam> a(10, 20);
    ASSERT_EQ(a.size(), 10);
    ASSERT_EQ(a.capacity(), 10);

    for (INDEX_TYPE i = 0; i < a.size(); ++i)
    {
      ASSERT_EQ(a.sizeOfArray(i), 0);
      ASSERT_EQ(a.capacityOfArray(i), 20);
    }
  }
}

TYPED_TEST(ArrayOfArraysTest, appendArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;

  ArrayOfArrays<TypeParam> array;
  REF_TYPE<TypeParam> ref;
  this->appendArray(array, ref, NUM_ARRAYS, MAX_VALUES);
}

TYPED_TEST(ArrayOfArraysTest, insertArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertArray(array, ref, NUM_ARRAYS, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, eraseArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendArray(array, ref, NUM_ARRAYS, MAX_VALUES);
    this->eraseArray(array, ref, array.size() / 2);
  }
}

TYPED_TEST(ArrayOfArraysTest, resize)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertArray(array, ref, NUM_ARRAYS, MAX_VALUES);
    this->resize(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysTest, resizeArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->resizeArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, appendToArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, appendMultipleToArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendMultipleToArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, insertIntoArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, insertMultipleIntoArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertMultipleIntoArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, eraseFromArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    this->eraseFromArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, removeMultipleFromArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    this->removeMultipleFromArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, compress)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    this->compress(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysTest, capacity)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->resizeArray(array, ref, MAX_VALUES);
    this->fill(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysTest, capacityOfArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 30;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->capacityOfArray(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysTest, deepCopy)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 30;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  this->appendToArray(array, ref, MAX_VALUES);
  this->deepCopy(array, ref);
}

TYPED_TEST(ArrayOfArraysTest, shallowCopy)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 30;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  this->appendToArray(array, ref, MAX_VALUES);
  this->shallowCopy(array);
}

#ifdef USE_CUDA

template <class T>
using Array1D = Array<T, 1, INDEX_TYPE>;

template <class T>
using Array2D = Array<T, 2, INDEX_TYPE>;

template <class T>
class ArrayOfArraysCudaTest : public ArrayOfArraysTest<T>
{
public:

  void memoryMotion(ViewType<T, true> const & view, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(view.toViewCC(), ref);

    INDEX_TYPE const nArrays = view.size();
    
    // Update the view on the device.
    forall(cuda(), 0, nArrays,
      [view] __device__ (INDEX_TYPE i)
      {
        INDEX_TYPE const sizeOfArray = view.sizeOfArray(i);
        for (INDEX_TYPE j = 0; j < sizeOfArray; ++j)
        {
          GEOS_ERROR_IF(&(view[i][j]) != &(view(i, j)), "Value mismatch!");
          view[i][j] += T(1);
          view(i, j) += T(1);
        }
      }
    );

    // Update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      for (INDEX_TYPE j = 0; j < ref[i].size(); ++j)
      {
        ref[i][j] += T(1);
        ref[i][j] += T(1);
      }
    }

    // Move the view back to the host and compare with the reference.
    forall(sequential(), 0, 1,
      [view, &ref] (INDEX_TYPE)
      {
        COMPARE_TO_REFERENCE(view.toViewCC(), ref);
      }
    );
  }

  void memoryMotionMove(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    
    // Update the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toViewC()] __device__ (INDEX_TYPE i)
      {
        INDEX_TYPE const sizeOfArray = view.sizeOfArray(i);
        for (INDEX_TYPE j = 0; j < sizeOfArray; ++j)
        {
          GEOS_ERROR_IF(&(view[i][j]) != &(view(i, j)), "Value mismatch!");
          view[i][j] += T(1);
          view(i, j) += T(1);
        }
      }
    );

    // Update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      for (INDEX_TYPE j = 0; j < ref[i].size(); ++j)
      {
        ref[i][j] += T(1);
        ref[i][j] += T(1);
      }
    }

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);
  }

  void appendDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array1D<Array1D<T>> valuesToAppend(nArrays);

    // Populate the valuesToAppend array and append the values to the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE maxAppends = array.capacityOfArray(i) - arraySize;
      INDEX_TYPE const nValues = rand(0, maxAppends);
      
      valuesToAppend[i].resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);
        ref[i].push_back(value);
        valuesToAppend[i][j] = value;
      }
    }

    // Append to the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), toAppend=valuesToAppend.toViewConst()] __device__ (INDEX_TYPE i)
      {
        for (INDEX_TYPE j = 0; j < toAppend[i].size(); ++j)
        {
          view.appendToArray(i, toAppend[i][j]);
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);
  }

  void appendMultipleDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array1D<Array1D<T>> valuesToAppend(nArrays);

    // Populate the valuesToAppend array and update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const maxAppends = array.capacityOfArray(i) - arraySize;
      INDEX_TYPE const nValues = rand(0, maxAppends);
      valuesToAppend[i].resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);
        ref[i].push_back(value);
        valuesToAppend[i][j] = value;
      }
    }

    // Append to the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), toAppend=valuesToAppend.toViewConst()] __device__ (INDEX_TYPE i)
      {
        view.appendToArray(i, toAppend[i].data(), toAppend[i].size());
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);
  }

  void insertDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array1D<Array1D<T>> valuesToInsert(nArrays);
    Array1D<Array1D<INDEX_TYPE>> positionsToInsert(nArrays);

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const maxInserts = array.capacityOfArray(i) - arraySize;
      INDEX_TYPE const nValues = rand(0, maxInserts);
      valuesToInsert[i].resize(nValues);
      positionsToInsert[i].resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);

        INDEX_TYPE const insertPos = rand(0, arraySize + j);
        ref[i].insert(ref[i].begin() + insertPos, value);
        valuesToInsert[i][j] = value;
        positionsToInsert[i][j] = insertPos;
      }
    }

    // Insert into the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), positions=positionsToInsert.toViewConst(), values=valuesToInsert.toViewConst()]
      __device__ (INDEX_TYPE i)
      {
        for (INDEX_TYPE j = 0; j < values[i].size(); ++j)
        {
          view.insertIntoArray(i, positions[i][j], values[i][j]);
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);
  }

  void insertMultipleDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array1D<Array1D<T>> valuesToInsert(nArrays);
    Array1D<INDEX_TYPE> positionsToInsert(nArrays);

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const maxInserts = array.capacityOfArray(i) - arraySize;
      INDEX_TYPE const nValues = rand(0, maxInserts);
      valuesToInsert[i].resize(nValues);

      INDEX_TYPE const insertPos = rand(0, arraySize);
      positionsToInsert[i] = insertPos;

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(i * LARGE_NUMBER + arraySize + j);
        valuesToInsert[i][j] = value;
      }
    
      ref[i].insert(ref[i].begin() + insertPos, valuesToInsert[i].begin(), valuesToInsert[i].end());
    }

    // Insert into the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), positions=positionsToInsert.toViewConst(), values=valuesToInsert.toViewConst()] 
      __device__ (INDEX_TYPE i)
      {
        view.insertIntoArray(i, positions[i], values[i].data(), values[i].size());
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewCC(), ref);
  }

  void removeDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxRemoves)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array1D<Array1D<INDEX_TYPE>> positionsToRemove(nArrays);

    // Populate the positionsToRemove array and update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      INDEX_TYPE const nToRemove = rand(0, min(arraySize, maxRemoves));
      positionsToRemove[i].resize(nToRemove);

      for (INDEX_TYPE j = 0; j < nToRemove; ++j)
      {
        INDEX_TYPE const removePosition = rand(0, arraySize - j - 1);
        ref[i].erase(ref[i].begin() + removePosition);
        positionsToRemove[i][j] = removePosition;
      }
    }

    // Remove the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), positions=positionsToRemove.toViewConst()]
      __device__ (INDEX_TYPE i)
      {
        for (INDEX_TYPE j = 0; j < positions[i].size(); ++j)
        {
          view.eraseFromArray(i, positions[i][j]);
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    compareToReference(array.toViewCC(), ref);
  }

  void removeMultipleDevice(ArrayOfArrays<T> & array, REF_TYPE<T> & ref, INDEX_TYPE const maxRemoves)
  {
    compareToReference(array.toViewCC(), ref);

    INDEX_TYPE const nArrays = array.size();
    Array2D<INDEX_TYPE> removals(nArrays, 2);

    // Populate the removals array and update the reference.
    for (INDEX_TYPE i = 0; i < nArrays; ++i)
    {
      INDEX_TYPE const arraySize = array.sizeOfArray(i);
      if (arraySize == 0) continue;

      INDEX_TYPE const removePosition = rand(0, arraySize - 1);
      INDEX_TYPE const nToRemove = rand(0, min(maxRemoves, arraySize - removePosition));

      removals[i][0] = removePosition;
      removals[i][1] = nToRemove;
      ref[i].erase(ref[i].begin() + removePosition, ref[i].begin() + removePosition + nToRemove);
    }

    // Remove from the view on the device.
    forall(cuda(), 0, nArrays,
      [view=array.toView(), removals=removals.toViewConst()]
      __device__ (INDEX_TYPE i)
      {
        if (view.sizeOfArray(i) == 0) return;
        view.eraseFromArray(i, removals[i][0], removals[i][1]);
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    compareToReference(array.toViewCC(), ref);
  }

protected:
  using ArrayOfArraysTest<T>::rand;
  using ArrayOfArraysTest<T>::LARGE_NUMBER;
  using ArrayOfArraysTest<T>::m_gen;
};

using CudaTestTypes = ::testing::Types<INDEX_TYPE, Tensor>;
TYPED_TEST_CASE(ArrayOfArraysCudaTest, CudaTestTypes);

TYPED_TEST(ArrayOfArraysCudaTest, memoryMotion)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 30;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->memoryMotion(array.toViewC(), ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, memoryMotionMove)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 30;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->memoryMotionMove(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, append)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->appendDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, appendMultiple)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->appendMultipleDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, insert)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->insertDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, insertMultiple)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->insertMultipleDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, remove)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 100;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->removeDevice(array, ref, MAX_VALUES);
  }
}

TYPED_TEST(ArrayOfArraysCudaTest, removeMultiple)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 10;
  constexpr INDEX_TYPE MAX_VALUES = 10;
  constexpr INDEX_TYPE N_ITER = 3;

  ArrayOfArrays<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendToArray(array, ref, MAX_VALUES);
    array.registerTouch(chai::CPU);
    this->removeMultipleDevice(array, ref, MAX_VALUES);
  }
}

#endif // USE_CUDA

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
