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

#include "ArrayOfSets.hpp"
#include "ArrayOfArrays.hpp"
#include "testUtils.hpp"

#ifdef USE_CUDA
#include "Array.hpp"
#include "SortedArray.hpp"
#endif

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#include <vector>
#include <random>

namespace LvArray
{

using INDEX_TYPE = std::ptrdiff_t;

template <class T>
using ViewType = ArrayOfSetsView<T, INDEX_TYPE const>;

template <class T>
using REF_TYPE = std::vector<std::set<T>>;

template <class T>
void compareToReference(ViewType<T const> const & view, REF_TYPE<T> const & ref)
{
  view.consistencyCheck();

  ASSERT_EQ(view.size(), ref.size());
  
  for (INDEX_TYPE i = 0; i < view.size(); ++i)
  {
    ASSERT_EQ(view.sizeOfSet(i), ref[i].size());

    typename std::set<T>::iterator it = ref[i].begin();
    for (INDEX_TYPE j = 0; j < view.sizeOfSet(i); ++j)
    {
      EXPECT_EQ(view[i][j], *it++);
      EXPECT_EQ(view(i, j), view[i][j]);
    }
    EXPECT_EQ(it, ref[i].end());

    it = ref[i].begin();
    for (T const & val : view.getIterableSet(i))
    {
      EXPECT_EQ(val, *it++);
      EXPECT_TRUE(view.contains(i, val));
    }
    EXPECT_EQ(it, ref[i].end());
  }
}

// Use this macro to call compareToReference with better error messages.
#define COMPARE_TO_REFERENCE( view, ref ) { SCOPED_TRACE( "" ); compareToReference( view, ref ); }

template <class T>
void printArray(ViewType<T const> const & view)
{
  std::cout << "{" << std::endl;

  for (INDEX_TYPE i = 0; i < view.size(); ++i)
  {
    std::cout << "\t{";
    for (INDEX_TYPE j = 0; j < view.sizeOfSet(i); ++j)
    {
      std::cout << view[i][j] << ", ";
    }

    for (INDEX_TYPE j = view.sizeOfSet(i); j < view.capacityOfSet(i); ++j)
    {
      std::cout << "X" << ", ";
    }

    std::cout << "}" << std::endl;
  }

  std::cout << "}" << std::endl;
}

template <class T>
INDEX_TYPE insertIntoRef(std::set<T> & ref, std::vector<T> const & values)
{
  INDEX_TYPE numInserted = 0;
  for (T const & val : values)
  {
    numInserted += ref.insert(val).second;
  }

  return numInserted;
}

template <class T>
class ArrayOfSetsTest : public ::testing::Test
{
public:

  void appendSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                 INDEX_TYPE const nSets, INDEX_TYPE const maxInserts,
                 INDEX_TYPE const maxValue)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    std::vector<T> arrayToAppend(maxInserts);

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);
      arrayToAppend.resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        arrayToAppend[j] = T(rand(0, maxValue));
      }

      array.appendSet(nValues);
      EXPECT_EQ(nValues, array.capacityOfSet(array.size() - 1));
      array.insertIntoSet(array.size() - 1, arrayToAppend.data(), arrayToAppend.size());
      ref.push_back(std::set<T>(arrayToAppend.begin(), arrayToAppend.end()));
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void insertSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                 INDEX_TYPE const nSets, INDEX_TYPE const maxInserts,
                 INDEX_TYPE const maxValue)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    std::vector<T> arrayToAppend(maxInserts);

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);
      arrayToAppend.resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        arrayToAppend[j] = T(rand(0, maxValue));
      }

      INDEX_TYPE insertPos = rand(0, array.size());

      array.insertSet(insertPos, nValues);
      EXPECT_EQ(nValues, array.capacityOfSet(insertPos));

      array.insertIntoSet(insertPos, arrayToAppend.data(), arrayToAppend.size());
      ref.insert(ref.begin() + insertPos, std::set<T>(arrayToAppend.begin(), arrayToAppend.end()));
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void eraseSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                INDEX_TYPE const nSets)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const removePos = rand(0, array.size() -1);
      array.eraseSet(removePos);
      ref.erase(ref.begin() + removePos);
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void resize(ArrayOfSets<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const newSize = rand(0, 2 * array.size());
    array.resize(newSize);
    ref.resize(newSize);

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void insertIntoSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                     INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(rand(0, maxValue));
        EXPECT_EQ(array.insertIntoSet(i, value), ref[i].insert(value).second);
      }
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void insertMultipleIntoSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                             INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue,
                             bool const sorted)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    std::vector<T> valuesToInsert(maxInserts);
    INDEX_TYPE const nSets = array.size();

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);
      valuesToInsert.resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        valuesToInsert[j] = T(rand(0, maxValue));
      }

      INDEX_TYPE numInserted;
      if (sorted)
      {
        std::sort(valuesToInsert.begin(), valuesToInsert.end());
        numInserted = array.insertSortedIntoSet(i, valuesToInsert.data(), valuesToInsert.size());
      }
      else
      {
        numInserted = array.insertIntoSet(i, valuesToInsert.data(), valuesToInsert.size());
      }

      EXPECT_EQ(numInserted, insertIntoRef(ref[i], valuesToInsert));
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void removeFromSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                     INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(rand(0, maxValue));
        EXPECT_EQ(array.removeFromSet(i, value), ref[i].erase(value));
      }
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void removeMultipleFromSet(ArrayOfSets<T> & array, REF_TYPE<T> & ref,
                             INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue,
                             bool const sorted)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    std::vector<T> valuesToRemove(maxInserts);
    INDEX_TYPE const nSets = array.size();

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);
      valuesToRemove.resize(nValues);

      INDEX_TYPE numRemovedFromRef = 0;
      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value(rand(0, maxValue));
        valuesToRemove[j] = value;
        numRemovedFromRef += ref[i].erase(value);
      }

      if (sorted)
      {
        std::sort(valuesToRemove.begin(), valuesToRemove.end());
        EXPECT_EQ(array.removeSortedFromSet(i, valuesToRemove.data(), valuesToRemove.size()), numRemovedFromRef);
      }
      else
      {
        EXPECT_EQ(array.removeFromSet(i, valuesToRemove.data(), valuesToRemove.size()), numRemovedFromRef);
      }
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void compress(ArrayOfSets<T> & array, REF_TYPE<T> const & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    array.compress();
    T const * const values = array[0];

    INDEX_TYPE curOffset = 0;
    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      EXPECT_EQ(array.sizeOfSet(i), array.capacityOfSet(i));

      T const * const curValues = array[i];
      EXPECT_EQ(values + curOffset, curValues);

      curOffset += array.sizeOfSet(i);
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void reserveSet(ArrayOfSets<T> & array, INDEX_TYPE const maxIncrease)
  {
    INDEX_TYPE const nSets = array.size();
    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const increment = rand(-maxIncrease, maxIncrease);
      INDEX_TYPE const newCapacity = std::max(INDEX_TYPE(0), array.sizeOfSet(i) + increment);
      
      array.reserveCapacityOfSet(i, newCapacity);
      ASSERT_GE(array.capacityOfSet(i), newCapacity);
    }
  }

  void fill(ArrayOfSets<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    T const * const values = array[0];
    T const * const endValues = array[array.size() - 1];

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      fillSet(array, ref, i);
    }

    T const * const newValues = array[0];
    T const * const newEndValues = array[array.size() - 1];
    EXPECT_EQ(values, newValues);
    EXPECT_EQ(endValues, newEndValues);

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void deepCopy(ArrayOfSets<T> const & array, REF_TYPE<T> const & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    ArrayOfSets<T> copy(array);

    EXPECT_EQ(array.size(), copy.size());

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      INDEX_TYPE const setSize = array.sizeOfSet(i);
      EXPECT_EQ(setSize, copy.sizeOfSet(i));
      
      T const * const values = array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_NE(values, valuesCopy);

      for (INDEX_TYPE j = setSize -1; j >= 0; --j)
      {
        T const val = array(i, j);
        EXPECT_EQ(val, copy(i, j));
        copy.removeFromSet(i, val);
      }

      EXPECT_EQ(copy.sizeOfSet(i), 0);
      EXPECT_EQ(array.sizeOfSet(i), setSize);
    }

    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void shallowCopy(ArrayOfSets<T> const & array)
  {
    ViewType<T> const copy(array.toView());

    EXPECT_EQ(array.size(), copy.size());

    for (INDEX_TYPE i = 0; i < array.size(); ++i)
    {
      INDEX_TYPE const setSize = array.sizeOfSet(i);
      EXPECT_EQ(setSize, copy.sizeOfSet(i));

      T const * const values = array[i];
      T const * const valuesCopy = copy[i];
      EXPECT_EQ(values, valuesCopy);

      for (INDEX_TYPE j = setSize -1; j >= 0; --j)
      {
        EXPECT_EQ(j, array.sizeOfSet(i) - 1);
        EXPECT_EQ(j, copy.sizeOfSet(i) - 1);

        T const val = array(i, j);
        EXPECT_EQ(val, copy(i, j));
        copy.removeFromSet(i, val);
      }

      EXPECT_EQ(copy.sizeOfSet(i), 0);
      EXPECT_EQ(array.sizeOfSet(i), 0);
    }
  }

  void stealFrom(ArrayOfSets<T> & array,
                 REF_TYPE<T> & ref,
                 INDEX_TYPE const maxValue,
                 INDEX_TYPE const maxInserts,
                 sortedArrayManipulation::Description const desc)
  {
    INDEX_TYPE const nSets = array.size();
    ArrayOfArrays<T> arrayToSteal(nSets);

    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const nValues = rand(0, maxInserts);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(rand(0, maxValue));
        bool const insertSuccess = ref[i].insert(value).second;

        if (sortedArrayManipulation::isUnique(desc))
        {
          if (insertSuccess)
          {
            arrayToSteal.appendToArray(i, value);
          }
        }
        else
        {
          arrayToSteal.appendToArray(i, value);
        }
      }

      if (sortedArrayManipulation::isSorted(desc))
      {
        T * const values = arrayToSteal[i];
        std::sort(values, values + arrayToSteal.sizeOfArray(i));
      }
    }

    array.stealFrom(std::move(arrayToSteal), desc);
  }

protected:

  void fillSet(ViewType<T> const & array, REF_TYPE<T> & ref, INDEX_TYPE const i)
  {
    INDEX_TYPE const setSize = array.sizeOfSet(i);
    INDEX_TYPE const setCapacity = array.capacityOfSet(i);
    INDEX_TYPE nToInsert = setCapacity - setSize;
    
    T const * const values = array[i];
    T const * const endValues = array[array.size() - 1];

    while (nToInsert > 0)
    {
      T const testValue = T(rand(0, 2 * setCapacity));
      bool const inserted = array.insertIntoSet(i, testValue);
      EXPECT_EQ(inserted, ref[i].insert(testValue).second);
      nToInsert -= inserted;
    }

    T const * const newValues = array[i];
    T const * const newEndValues = array[array.size() - 1];
    INDEX_TYPE const endCapacity = array.capacityOfSet(i);
    ASSERT_EQ(values, newValues);
    ASSERT_EQ(endValues, newEndValues);
    ASSERT_EQ(setCapacity, endCapacity);
  }

  INDEX_TYPE rand(INDEX_TYPE const min, INDEX_TYPE const max)
  {
    return std::uniform_int_distribution<INDEX_TYPE>(min, max)(m_gen);
  }

  static constexpr INDEX_TYPE LARGE_NUMBER = 1E6;
  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types<INDEX_TYPE, Tensor, TestString>;
TYPED_TEST_CASE(ArrayOfSetsTest, TestTypes);

TYPED_TEST(ArrayOfSetsTest, construction)
{
  // Test empty construction
  {
    ArrayOfSets<TypeParam> a;
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(a.capacity(), 0);
  }

  // Test construction with number of arrays
  {
    ArrayOfSets<TypeParam> a(10);
    ASSERT_EQ(a.size(), 10);
    ASSERT_EQ(a.capacity(), 10);

    for (INDEX_TYPE i = 0; i < a.size(); ++i)
    {
      ASSERT_EQ(a.sizeOfSet(i), 0);
      ASSERT_EQ(a.capacityOfSet(i), 0);
    }
  }

  // Test construction with a number of arrays and a hint
  {
    ArrayOfSets<TypeParam> a(10, 20);
    ASSERT_EQ(a.size(), 10);
    ASSERT_EQ(a.capacity(), 10);

    for (INDEX_TYPE i = 0; i < a.size(); ++i)
    {
      ASSERT_EQ(a.sizeOfSet(i), 0);
      ASSERT_EQ(a.capacityOfSet(i), 20);
    }
  }
}

TYPED_TEST(ArrayOfSetsTest, appendSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array;
  REF_TYPE<TypeParam> ref;
  this->appendSet(array, ref, NUM_ARRAYS, MAX_INSERTS, MAX_VALUE);
}

TYPED_TEST(ArrayOfSetsTest, insertSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertSet(array, ref, NUM_ARRAYS, MAX_INSERTS, MAX_VALUE);
  }
}

TYPED_TEST(ArrayOfSetsTest, eraseArray)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->appendSet(array, ref, NUM_ARRAYS, MAX_INSERTS, MAX_VALUE);
    this->eraseSet(array, ref, array.size() / 2);
  }
}

TYPED_TEST(ArrayOfSetsTest, resize)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array;
  REF_TYPE<TypeParam> ref;

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertSet(array, ref, NUM_ARRAYS, MAX_INSERTS, MAX_VALUE);
    this->resize(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsTest, insertIntoSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
  }
}

TYPED_TEST(ArrayOfSetsTest, insertMultipleIntoSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertMultipleIntoSet(array, ref, MAX_INSERTS, MAX_VALUE, false);
  }
}

TYPED_TEST(ArrayOfSetsTest, insertMultipleIntoSetSorted)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertMultipleIntoSet(array, ref, MAX_INSERTS, MAX_VALUE, true);
  }
}

TYPED_TEST(ArrayOfSetsTest, removeFromSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    this->removeFromSet(array, ref, MAX_INSERTS, MAX_VALUE);
  }
}

TYPED_TEST(ArrayOfSetsTest, removeMultipleFromSet)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    this->removeMultipleFromSet(array, ref, MAX_INSERTS, MAX_VALUE, false);
  }
}

TYPED_TEST(ArrayOfSetsTest, removeMultipleFromSetSorted)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    this->removeMultipleFromSet(array, ref, MAX_INSERTS, MAX_VALUE, true);
  }
}

TYPED_TEST(ArrayOfSetsTest, compress)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    this->compress(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsTest, capacity)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INCREASE = 50;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->reserveSet(array, MAX_INCREASE);
    this->fill(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsTest, deepCopy)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
  this->deepCopy(array, ref);
}

TYPED_TEST(ArrayOfSetsTest, shallowCopy)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
  this->shallowCopy(array);
}

TYPED_TEST(ArrayOfSetsTest, stealFromSortedUnique)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->stealFrom(array, ref, MAX_INSERTS, MAX_VALUE, sortedArrayManipulation::SORTED_UNIQUE);
  COMPARE_TO_REFERENCE(array.toViewC(), ref);
}

TYPED_TEST(ArrayOfSetsTest, stealFromUnsortedNoDuplicates)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->stealFrom(array, ref, MAX_INSERTS, MAX_VALUE, sortedArrayManipulation::UNSORTED_NO_DUPLICATES);
  COMPARE_TO_REFERENCE(array.toViewC(), ref);
}

TYPED_TEST(ArrayOfSetsTest, stealFromSortedWithDuplicates)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->stealFrom(array, ref, MAX_INSERTS, MAX_VALUE, sortedArrayManipulation::SORTED_WITH_DUPLICATES);
  COMPARE_TO_REFERENCE(array.toViewC(), ref);
}

TYPED_TEST(ArrayOfSetsTest, stealFromUnsortedWithDuplicates)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);

  this->stealFrom(array, ref, MAX_INSERTS, MAX_VALUE, sortedArrayManipulation::UNSORTED_NO_DUPLICATES);
  COMPARE_TO_REFERENCE(array.toViewC(), ref);
}

// Note this is testing capabilities of the ArrayOfArrays class, howver it needs to first populate
// the ArrayOfSets so it involves less code duplication to put it here.
TYPED_TEST(ArrayOfSetsTest, ArrayOfArraysStealFrom)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;

  ArrayOfArrays<TypeParam> array;
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  {
    ArrayOfSets<TypeParam> arrayToStealFrom(NUM_ARRAYS);
    this->insertIntoSet(arrayToStealFrom, ref, MAX_INSERTS, MAX_VALUE);
    array.stealFrom(std::move(arrayToStealFrom));
  }

  ASSERT_EQ(array.size(), ref.size());
  
  for (INDEX_TYPE i = 0; i < array.size(); ++i)
  {
    ASSERT_EQ(array.sizeOfArray(i), ref[i].size());

    typename std::set<TypeParam>::iterator it = ref[i].begin();
    for (INDEX_TYPE j = 0; j < array.sizeOfArray(i); ++j)
    {
      EXPECT_EQ(array[i][j], *it++);
    }
    EXPECT_EQ(it, ref[i].end());
  }
}

#ifdef USE_CUDA

template <class T>
using Array1D = Array<T, 1, RAJA::PERM_I, INDEX_TYPE>;

template <class T>
class ArrayOfSetsCudaTest : public ArrayOfSetsTest<T>
{
public:

  void memoryMotion(ViewType<T> const & view, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(view.toViewC(), ref);

    INDEX_TYPE const nSets = view.size();
    
    // Update the view on the device.
    forall(gpu(), 0, nSets,
      [view] __device__ (INDEX_TYPE i)
      {
        INDEX_TYPE const sizeOfSet = view.sizeOfSet(i);
        for (INDEX_TYPE j = 0; j < sizeOfSet; ++j)
        {
          LVARRAY_ERROR_IF(&(view[i][j]) != &(view(i, j)), "Value mismatch!");
        }

        for (T const & val : view.getIterableSet(i))
        {
          LVARRAY_ERROR_IF(!view.contains(i, val), "Set should contain the value");
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    forall(sequential(), 0, 1,
      [view, &ref] (INDEX_TYPE)
      {
        COMPARE_TO_REFERENCE(view.toViewC(), ref);
      }
    );
  }

  void memoryMotionMove(ArrayOfSets<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    
    // Update the view on the device.
    array.move(chai::GPU);
    forall(gpu(), 0, nSets,
      [view=array.toView()] __device__ (INDEX_TYPE i)
      {
        INDEX_TYPE const sizeOfSet = view.sizeOfSet(i);
        for (INDEX_TYPE j = 0; j < sizeOfSet; ++j)
        {
          LVARRAY_ERROR_IF(&(view[i][j]) != &(view(i, j)), "Value mismatch!");
        }

        for (T const & val : view.getIterableSet(i))
        {
          LVARRAY_ERROR_IF(!view.contains(i, val), "Set should contain the value");
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void insertDevice(ArrayOfSets<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    Array1D<Array1D<T>> valuesToInsert(nSets);

    // Populate the valuesToInsert array and append the values to the reference.
    populateValuesToInsert(array, ref, valuesToInsert, false);

    // Append to the view on the device.
    forall(gpu(), 0, nSets,
      [view=array.toView(), toInsert=valuesToInsert.toViewConst()] __device__ (INDEX_TYPE const i)
      {
        for (INDEX_TYPE j = 0; j < toInsert[i].size(); ++j)
        {
          view.insertIntoSet(i, toInsert[i][j]);
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void insertMultipleDevice(ArrayOfSets<T> & array, REF_TYPE<T> & ref, bool const sorted)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    Array1D<Array1D<T>> valuesToInsert(nSets);

    // Populate the valuesToInsert array and append the values to the reference.
    populateValuesToInsert(array, ref, valuesToInsert, sorted);

    // Append to the view on the device.
    if (sorted)
    {
      forall(gpu(), 0, nSets,
        [view=array.toView(), toInsert=valuesToInsert.toViewConst()] __device__ (INDEX_TYPE const i)
        {
          view.insertSortedIntoSet(i, toInsert[i].data(), toInsert[i].size());
        }
      );
    }
    else
    {
      forall(gpu(), 0, nSets,
        [view=array.toView(), toInsert=valuesToInsert.toViewConst()] __device__ (INDEX_TYPE const i)
        {
          view.insertIntoSet(i, toInsert[i].data(), toInsert[i].size());
        }
      );
    }

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void removeDevice(ArrayOfSets<T> & array, REF_TYPE<T> & ref)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    Array1D<Array1D<T>> valuesToRemove(nSets);

    // Populate the valuesToRemove array and remove the values from the reference.
    populateValuesToRemove(array, ref, valuesToRemove, false);

    // Append to the view on the device.
    forall(gpu(), 0, nSets,
      [view=array.toView(), toRemove=valuesToRemove.toViewConst()] __device__ (INDEX_TYPE const i)
      {
        for (INDEX_TYPE j = 0; j < toRemove[i].size(); ++j)
        {
          view.removeFromSet(i, toRemove[i][j]);
        }
      }
    );

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }

  void removeMultiple(ArrayOfSets<T> & array, REF_TYPE<T> & ref, bool const sorted)
  {
    COMPARE_TO_REFERENCE(array.toViewC(), ref);

    INDEX_TYPE const nSets = array.size();
    Array1D<Array1D<T>> valuesToRemove(nSets);

    // Populate the valuesToRemove array and remove the values from the reference.
    populateValuesToRemove(array, ref, valuesToRemove, sorted);

    // Append to the view on the device.
    if (sorted)
    {
      forall(gpu(), 0, nSets,
        [view=array.toView(), toRemove=valuesToRemove.toViewConst()] __device__ (INDEX_TYPE const i)
        {
          view.removeSortedFromSet(i, toRemove[i].data(), toRemove[i].size());
        }
      );
    }
    else
    {
      forall(gpu(), 0, nSets,
        [view=array.toView(), toRemove=valuesToRemove.toViewConst()] __device__ (INDEX_TYPE const i)
        {
          view.removeFromSet(i, toRemove[i].data(), toRemove[i].size());
        }
      );
    }

    // Move the view back to the host and compare with the reference.
    array.move(chai::CPU);
    COMPARE_TO_REFERENCE(array.toViewC(), ref);
  }
  
protected:

  void populateValuesToRemove(ViewType<T const> const & array,
                              REF_TYPE<T> & ref,
                              Array1D<Array1D<T>> & valuesToRemove,
                              bool const sorted)
  {
    INDEX_TYPE const nSets = array.size();
    LVARRAY_ERROR_IF_NE(nSets, valuesToRemove.size());

    // Populate the valuesToRemove array and remove the values from the reference.
    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const sizeOfSet = array.sizeOfSet(i);
      INDEX_TYPE const capacityOfSet = array.capacityOfSet(i);
      INDEX_TYPE const nValues = rand(0, capacityOfSet);
      
      valuesToRemove[i].resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(rand(0, 2 * capacityOfSet));
        ref[i].erase(value);
        valuesToRemove[i][j] = value;
      }

      if (sorted) std::sort(valuesToRemove[i].begin(), valuesToRemove[i].end());
    }
  }

  void populateValuesToInsert(ViewType<T const> const & array,
                              REF_TYPE<T> & ref,
                              Array1D<Array1D<T>> & valuesToInsert,
                              bool const sorted)
  {
    INDEX_TYPE const nSets = array.size();
    LVARRAY_ERROR_IF_NE(nSets, valuesToInsert.size());
    
    for (INDEX_TYPE i = 0; i < nSets; ++i)
    {
      INDEX_TYPE const sizeOfSet = array.sizeOfSet(i);
      INDEX_TYPE const capacityOfSet = array.capacityOfSet(i);
      INDEX_TYPE const nValues = rand(0, capacityOfSet - sizeOfSet);
      
      valuesToInsert[i].resize(nValues);

      for (INDEX_TYPE j = 0; j < nValues; ++j)
      {
        T const value = T(rand(0, 2 * capacityOfSet));
        ref[i].insert(value);
        valuesToInsert[i][j] = value;
      }

      if (sorted) std::sort(valuesToInsert[i].begin(), valuesToInsert[i].end());
    }
  }

  using ArrayOfSetsTest<T>::rand;
  using ArrayOfSetsTest<T>::LARGE_NUMBER;
  using ArrayOfSetsTest<T>::m_gen;
};

using CudaTestTypes = ::testing::Types<INDEX_TYPE, Tensor>;
TYPED_TEST_CASE(ArrayOfSetsCudaTest, CudaTestTypes);

TYPED_TEST(ArrayOfSetsCudaTest, memoryMotion)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->memoryMotion(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, memoryMotionMove)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->memoryMotionMove(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, insertDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->insertDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, insertMultipleDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->insertMultipleDevice(array, ref, false);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, insertSortedDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->insertMultipleDevice(array, ref, true);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, removeDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->removeDevice(array, ref);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, removeMultipleDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->removeMultiple(array, ref, false);
  }
}

TYPED_TEST(ArrayOfSetsCudaTest, removeSortedDevice)
{
  constexpr INDEX_TYPE NUM_ARRAYS = 50;
  constexpr INDEX_TYPE MAX_INSERTS = 50;
  constexpr INDEX_TYPE MAX_VALUE = 100;
  constexpr INDEX_TYPE N_ITER = 2;

  ArrayOfSets<TypeParam> array(NUM_ARRAYS);
  REF_TYPE<TypeParam> ref(NUM_ARRAYS);
  
  for (INDEX_TYPE i = 0; i < N_ITER; ++i)
  {
    this->insertIntoSet(array, ref, MAX_INSERTS, MAX_VALUE);
    array.registerTouch(chai::CPU);
    this->removeMultiple(array, ref, true);
  }
}

#endif // USE_CUDA

} // namespace LvArray
