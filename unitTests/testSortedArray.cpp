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

// Source includes
#include "SortedArray.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

#if defined(USE_CHAI) && defined(USE_CUDA)
  #include <chai/util/forall.hpp>
#endif

// System includes
#include <vector>
#include <set>
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

/**
 * @brief Check that the SortedArrayView is equivalent to the std::set. Checks equality using the
 *        operator[] and the raw pointer.
 * @param [in] set the SortedArrayView to check.
 * @param [in] ref the std::set to check against.
 */
template <class T>
void compareToReference(SortedArrayView<T const> const & set, const std::set<T> & ref )
{
  ASSERT_EQ( set.size(), ref.size() );
  ASSERT_EQ( set.empty(), ref.empty() );
  if( set.empty() )
  {
    ASSERT_EQ( set.size(), 0 );
    return;
  }

  T const * ptr = set.values();
  typename std::set< T >::const_iterator it = ref.begin();
  for(int i = 0 ; i < set.size(); ++i )
  {
    ASSERT_EQ( set[ i ], *it );
    ASSERT_EQ( ptr[ i ], *it );
    ++it;
  }
}

#define COMPARE_TO_REFERENCE( set, ref ) { SCOPED_TRACE( "" ); compareToReference( set, ref ); }

template< typename T >
class SortedArrayTest : public ::testing::Test
{
public:

  /**
   * @brief Test the insert method of the SortedArray.
   * @param [in] maxInserts the number of times to call insert.
   * @param [in] maxVal the largest value possibly generate.
   */
  void insertTest(INDEX_TYPE const maxInserts, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    for (INDEX_TYPE i = 0; i < maxInserts; ++i)
    {
      T const value = randVal(maxVal);
      ASSERT_EQ(m_ref.insert(value).second, m_set.insert(value));
    }

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the insert multiple sorted method of the SortedArray.
   * @param [in] maxInserts the number of values to insert at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void insertMultipleSortedTest(INDEX_TYPE const maxInserts, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    std::vector<T> values(maxInserts);


    for (INDEX_TYPE i = 0; i < maxInserts; ++i)
    {
      values[i] = randVal(maxVal);
    }

    std::sort(values.begin(), values.end());
    m_set.insertSorted(values.data(), maxInserts);
    m_ref.insert(values.begin(), values.end());

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the insert multiple method of the SortedArray.
   * @param [in] maxInserts the number of values to insert at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void insertMultipleTest(INDEX_TYPE const maxInserts, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    std::vector<T> values(maxInserts);

    for (INDEX_TYPE i = 0; i < maxInserts; ++i)
    {
      values[i] = randVal(maxVal);
    }

    m_set.insert(values.data(), maxInserts);
    m_ref.insert(values.begin(), values.end());

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the erase method of the SortedArray.
   * @param [in] maxRemoves the number of times to call erase.
   * @param [in] maxVal the largest value possibly generate.
   */
  void eraseTest(INDEX_TYPE const maxRemoves, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    for (INDEX_TYPE i = 0; i < maxRemoves; ++i)
    {
      T const value = randVal(maxVal);
      ASSERT_EQ(m_ref.erase(value), m_set.erase(value));
    }

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the erase multiple sorted method of the SortedArray.
   * @param [in] maxInserts the number of values to erase at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void eraseMultipleSortedTest(INDEX_TYPE const maxRemoves, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    std::vector<T> values(maxRemoves);

    for (INDEX_TYPE i = 0; i < maxRemoves; ++i)
    {
      values[i] = randVal(maxVal);
    }

    std::sort(values.begin(), values.end());
    m_set.eraseSorted(values.data(), maxRemoves);

    for (INDEX_TYPE i = 0; i < maxRemoves; ++i)
    {
      m_ref.erase(values[i]);
    }

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the erase multiple method of the SortedArray.
   * @param [in] maxInserts the number of values to erase at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void eraseMultipleTest(INDEX_TYPE const maxRemoves, INDEX_TYPE const maxVal)
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    std::vector<T> values(maxRemoves);

    for (INDEX_TYPE i = 0; i < maxRemoves; ++i)
    {
      values[i] = randVal(maxVal);
    }

    m_set.erase(values.data(), maxRemoves);

    for (INDEX_TYPE i = 0; i < maxRemoves; ++i)
    {
      m_ref.erase(values[i]);
    }

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

  /**
   * @brief Test the contains method of the SortedArray.
   */
  void containsTest() const
  {
    INDEX_TYPE const N = m_set.size();
    ASSERT_EQ(N, m_ref.size());

    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);

    typename std::set<T>::iterator it = m_ref.begin();
    for (INDEX_TYPE i = 0; i < N; ++i)
    {
      T const & value = *it;
      EXPECT_TRUE(m_set.contains(value));
      EXPECT_EQ(m_set.contains(T(i)), m_ref.count(T(i)));
      ++it;
    }
  }

  /**
   * @brief Test the copy constructor of the SortedArray.
   */
  void deepCopyTest() const
  {
    INDEX_TYPE const N = m_set.size();

    SortedArray<T> v_cpy(m_set);
    ASSERT_EQ(N, v_cpy.size());

    T const * const vals = m_set.values();
    T const * const vals_cpy = v_cpy.values();
    ASSERT_NE(vals, vals_cpy);

    // Iterate backwards and erase entries from v_cpy.
    for (INDEX_TYPE i = N - 1; i >= 0; --i)
    {
      EXPECT_EQ(vals[i], vals_cpy[i]);
      v_cpy.erase(vals_cpy[i]);
    }

    EXPECT_EQ(v_cpy.size(), 0);
    EXPECT_EQ(m_set.size(), N);

    // check that v is still the same as m_ref.
    COMPARE_TO_REFERENCE(m_set.toView(), m_ref);
  }

protected:

  T randVal(INDEX_TYPE const max)
  {
    return T( std::uniform_int_distribution<INDEX_TYPE>(0, max)(m_gen) );
  }

  SortedArray< T > m_set;

  std::set< T > m_ref;

 std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types<int, Tensor, TestString>;
TYPED_TEST_CASE(SortedArrayTest, TestTypes);

INDEX_TYPE const DEFAULT_MAX_INSERTS = 200;
INDEX_TYPE const DEFAULT_MAX_VAL = 1000;

TYPED_TEST(SortedArrayTest, insert)
{
  for ( int i = 0; i < 4; ++i )
  {
    this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, reserve)
{
  this->m_set.reserve( DEFAULT_MAX_VAL + 1 );
  TypeParam const * const ptr = this->m_set.values();

  for ( int i = 0; i < 4; ++i )
  {
    this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }

  EXPECT_EQ(ptr, this->m_set.values());
}

TYPED_TEST(SortedArrayTest, insertMultipleSorted)
{
  for ( int i = 0; i < 4; ++i )
  {
    this->insertMultipleSortedTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, insertMultiple)
{
  for ( int i = 0; i < 4; ++i )
  {
    this->insertMultipleTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, erase)
{
  for ( int i = 0; i < 2; ++i )
  {
    this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
    this->eraseTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, eraseMultipleSorted)
{
  for ( int i = 0; i < 2; ++i )
  {
    this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
    this->eraseMultipleSortedTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, eraseMultiple)
{
  for ( int i = 0; i < 2; ++i )
  {
    this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
    this->eraseMultipleTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  }
}

TYPED_TEST(SortedArrayTest, contains)
{
  this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  this->containsTest();
}

TYPED_TEST(SortedArrayTest, deepCopy)
{
  this->insertTest(DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL);
  this->deepCopyTest();
}

#ifdef USE_CUDA

template< typename T >
class SortedArrayCudaTest : public SortedArrayTest< T >
{
public:

  /**
   * @brief Test the SortedArrayView copy constructor in regards to memory motion.
   * @param [in] size the number of values to insert.
   */
  void memoryMotionTest(INDEX_TYPE const size)
  {
    ASSERT_TRUE(m_set.empty());

    // Insert values.
    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      m_set.insert(T(i));
    }

    ASSERT_EQ(m_set.size(), size);

    // Capture a view on the device.
    forall(gpu(), 0, size,
      [view = m_set.toView()] __device__ (INDEX_TYPE i)
      {
        LVARRAY_ERROR_IF(view[i] != T(i), "Values changed when moved.");
      }
    );

    // Change the values.
    m_set.clear();
    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      m_set.insert(T(i * i));
    }

    ASSERT_EQ(m_set.size(), size);

    // Capture the view on host and check that the values haven't been overwritten.
    forall(sequential(), 0, size,
      [view = m_set.toView()](INDEX_TYPE i)
      {
        EXPECT_EQ(view[i], T(i * i));
      }
    );
  }

  /**
   * @brief Test the SortedArray move method.
   * @param [in] size the number of values to insert.
   */
  void memoryMotionMoveTest(INDEX_TYPE const size)
  {
    ASSERT_TRUE(m_set.empty());

    // Insert values.
    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      m_set.insert(T(i));
    }

    ASSERT_EQ(m_set.size(), size);

    // Capture a view on the device.
    forall(gpu(), 0, size,
      [view = m_set.toView()] __device__ (INDEX_TYPE i)
      {
        LVARRAY_ERROR_IF(view[i] != T(i), "Values changed when moved.");
      }
    );

    // Change the values.
    m_set.clear();
    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      m_set.insert(T(i * i));
    }

    ASSERT_EQ(m_set.size(), size);

    // Move the array back to the host and check that the values haven't been overwritten.
    m_set.move(chai::CPU);
    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      EXPECT_EQ(m_set[i], T(i * i));
    }
  }

  /**
   * @brief Test the contains method of the SortedArray on device.
   * @param [in] size the number of values to insert.
   */
  void containsDeviceTest(INDEX_TYPE const size)
  {
    ASSERT_TRUE(m_set.empty());

    for (INDEX_TYPE i = 0; i < size; ++i)
    {
      m_set.insert(T(2 * i));
    }

    ASSERT_EQ(m_set.size(), size);

    forall(gpu(), 0, size,
      [view = m_set.toView()] __device__ (INDEX_TYPE i)
      {
        LVARRAY_ERROR_IF(!view.contains(T(2 * i)), "view should contain even numbers.");
        LVARRAY_ERROR_IF(view.contains(T(2 * i + 1)), "view should not contain odd numbers.");
      }
    );
  }

protected:
  using SortedArrayTest< T >::m_set;
  using SortedArrayTest< T >::m_ref;
};

using CudaTestTypes = ::testing::Types<int, Tensor>;
TYPED_TEST_CASE(SortedArrayCudaTest, CudaTestTypes);

TYPED_TEST(SortedArrayCudaTest, memoryMotion)
{
  this->memoryMotionTest(1000);
}

TYPED_TEST(SortedArrayCudaTest, memoryMotionMove)
{
  this->memoryMotionMoveTest(1000);
}

TYPED_TEST(SortedArrayCudaTest, containsDevice)
{
  this->containsDeviceTest(1000);
}

#endif

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
