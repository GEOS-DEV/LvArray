/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "typeManipulation.hpp"
#include "SortedArray.hpp"
#include "testUtils.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <vector>
#include <set>
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< typename SORTED_ARRAY >
class SortedArrayTest : public ::testing::Test
{
public:

  using T = typename SORTED_ARRAY::value_type;

  /**
   * @brief Check that the SortedArrayView is equivalent to the std::set. Checks equality using the
   *   operator[] and the raw pointer.
   */
  void compareToReference() const
  {
    EXPECT_EQ( m_set.size(), m_ref.size() );
    EXPECT_EQ( m_set.empty(), m_ref.empty() );
    if( m_set.empty() )
    {
      EXPECT_EQ( m_set.size(), 0 );
      return;
    }

    T const * ptr = m_set.data();
    ArraySlice< T const, 1, 0, INDEX_TYPE > const slice = m_set.toSlice();
    typename std::set< T >::const_iterator it = m_ref.begin();
    for( int i = 0; i < m_set.size(); ++i )
    {
      EXPECT_EQ( m_set[ i ], *it );
      EXPECT_EQ( m_set( i ), *it );
      EXPECT_EQ( ptr[ i ], *it );
      EXPECT_EQ( slice[ i ], *it );
      ++it;
    }

    it = m_ref.begin();
    for( T const & val : m_set )
    {
      EXPECT_EQ( val, *it );
      ++it;
    }

    EXPECT_EQ( it, m_ref.end() );
  }

  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); compareToReference(); \
  }

  /**
   * @brief Test the insert method of the SortedArray.
   * @param [in] maxInserts the number of times to call insert.
   * @param [in] maxVal the largest value possibly generate.
   */
  void insertTest( INDEX_TYPE const maxInserts, INDEX_TYPE const maxVal )
  {
    COMPARE_TO_REFERENCE

    for( INDEX_TYPE i = 0; i < maxInserts; ++i )
    {
      T const value = randVal( maxVal );
      ASSERT_EQ( m_ref.insert( value ).second, m_set.insert( value ));
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the insert multiple sorted method of the SortedArray.
   * @param [in] maxInserts the number of values to insert at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void insertMultipleTest( INDEX_TYPE maxInserts, INDEX_TYPE const maxVal )
  {
    COMPARE_TO_REFERENCE

    // Divide maxInserts by two since we insert with both a std::vector and std::set.
      maxInserts /= 2;

    // Insert values from a std::vector
    {
      std::vector< T > values( maxInserts );

      for( INDEX_TYPE i = 0; i < maxInserts; ++i )
      { values[ i ] = randVal( maxVal ); }

      m_ref.insert( values.begin(), values.end() );

      INDEX_TYPE const numUniqueValues = sortedArrayManipulation::makeSortedUnique( values.begin(), values.end() );
      values.resize( numUniqueValues );

      m_set.insert( values.begin(), values.end() );
      COMPARE_TO_REFERENCE
    }

    // Insert values from a std::set
    {
      std::set< T > values;
      for( INDEX_TYPE i = 0; i < maxInserts; ++i )
      { values.insert( randVal( maxVal ) ); }

      m_ref.insert( values.begin(), values.end() );
      m_set.insert( values.begin(), values.end() );
      COMPARE_TO_REFERENCE
    }
  }

  /**
   * @brief Test the remove method of the SortedArray.
   * @param [in] maxRemoves the number of times to call remove.
   * @param [in] maxVal the largest value possibly generate.
   */
  void removeTest( INDEX_TYPE const maxRemoves, INDEX_TYPE const maxVal )
  {
    COMPARE_TO_REFERENCE

    for( INDEX_TYPE i = 0; i < maxRemoves; ++i )
    {
      T const value = randVal( maxVal );
      ASSERT_EQ( m_ref.erase( value ), m_set.remove( value ) );
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the erase multiple sorted method of the SortedArray.
   * @param [in] maxInserts the number of values to erase at a time.
   * @param [in] maxVal the largest value possibly generate.
   */
  void removeMultipleTest( INDEX_TYPE maxRemoves, INDEX_TYPE const maxVal )
  {
    COMPARE_TO_REFERENCE

    // Divide maxInserts by two since we insert with both a std::vector and std::set.
      maxRemoves /= 2;

    // Remove values from a std::vector
    {
      std::vector< T > values( maxRemoves );

      for( INDEX_TYPE i = 0; i < maxRemoves; ++i )
      {
        values[ i ] = randVal( maxVal );
        m_ref.erase( values[ i ] );
      }

      INDEX_TYPE const numUniqueValues = sortedArrayManipulation::makeSortedUnique( values.begin(), values.end() );
      values.resize( numUniqueValues );

      m_set.remove( values.begin(), values.end() );

      COMPARE_TO_REFERENCE
    }

    // Remove values from a std::set
    {
      std::set< T > values;
      for( INDEX_TYPE i = 0; i < maxRemoves; ++i )
      {
        T const val = randVal( maxVal );
        m_ref.erase( val );
        values.insert( val );
      }

      m_set.remove( values.begin(), values.end() );

      COMPARE_TO_REFERENCE
    }
  }

  /**
   * @brief Test the contains method of the SortedArray.
   */
  void containsTest() const
  {
    COMPARE_TO_REFERENCE

    auto it = m_ref.begin();
    for( INDEX_TYPE i = 0; i < m_set.size(); ++i )
    {
      T const & value = *it;
      EXPECT_TRUE( m_set.contains( value ) );
      EXPECT_EQ( m_set.contains( T( i ) ), m_ref.count( T( i ) ) );
      ++it;
    }
  }

  /**
   * @brief Test the copy constructor of the SortedArray.
   */
  void deepCopyTest() const
  {
    COMPARE_TO_REFERENCE

    INDEX_TYPE const N = m_set.size();

    SORTED_ARRAY v_cpy( m_set );
    ASSERT_EQ( N, v_cpy.size());

    T const * const vals = m_set.data();
    T const * const vals_cpy = v_cpy.data();
    ASSERT_NE( vals, vals_cpy );

    // Iterate backwards and erase entries from v_cpy.
    for( INDEX_TYPE i = N - 1; i >= 0; --i )
    {
      EXPECT_EQ( vals[ i ], vals_cpy[ i ] );
      v_cpy.remove( vals_cpy[ i ] );
    }

    EXPECT_EQ( v_cpy.size(), 0 );
    EXPECT_EQ( m_set.size(), N );

    // check that v is still the same as m_ref.
    COMPARE_TO_REFERENCE
  }

protected:

  T randVal( INDEX_TYPE const max )
  { return T( std::uniform_int_distribution< INDEX_TYPE >( 0, max )( m_gen ) ); }

  SORTED_ARRAY m_set;

  std::set< T > m_ref;

  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types<
  SortedArray< int, INDEX_TYPE, MallocBuffer >
  , SortedArray< Tensor, INDEX_TYPE, MallocBuffer >
  , SortedArray< TestString, INDEX_TYPE, MallocBuffer >
#if defined(LVARRAY_USE_CHAI)
  , SortedArray< int, INDEX_TYPE, ChaiBuffer >
  , SortedArray< Tensor, INDEX_TYPE, ChaiBuffer >
  , SortedArray< TestString, INDEX_TYPE, ChaiBuffer >
#endif
  >;
TYPED_TEST_SUITE( SortedArrayTest, TestTypes, );

INDEX_TYPE const DEFAULT_MAX_INSERTS = 200;
INDEX_TYPE const DEFAULT_MAX_VAL = 1000;

TYPED_TEST( SortedArrayTest, insert )
{
  for( int i = 0; i < 4; ++i )
  {
    this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  }
}

TYPED_TEST( SortedArrayTest, reserve )
{
  this->m_set.reserve( DEFAULT_MAX_VAL + 1 );
  auto const * const ptr = this->m_set.data();

  for( int i = 0; i < 4; ++i )
  {
    this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  }

  EXPECT_EQ( ptr, this->m_set.data());
}

TYPED_TEST( SortedArrayTest, insertMultiple )
{
  for( int i = 0; i < 4; ++i )
  {
    this->insertMultipleTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  }
}

TYPED_TEST( SortedArrayTest, erase )
{
  for( int i = 0; i < 2; ++i )
  {
    this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
    this->removeTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  }
}

TYPED_TEST( SortedArrayTest, removeMultiple )
{
  for( int i = 0; i < 2; ++i )
  {
    this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
    this->removeMultipleTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  }
}

TYPED_TEST( SortedArrayTest, contains )
{
  this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  this->containsTest();
}

TYPED_TEST( SortedArrayTest, deepCopy )
{
  this->insertTest( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VAL );
  this->deepCopyTest();
}


template< typename SORTED_ARRAY_POLICY_PAIR >
class SortedArrayViewTest : public SortedArrayTest< typename SORTED_ARRAY_POLICY_PAIR::first_type >
{
public:

  using SORTED_ARRAY = typename SORTED_ARRAY_POLICY_PAIR::first_type;
  using POLICY = typename SORTED_ARRAY_POLICY_PAIR::second_type;

  using T = typename SORTED_ARRAY::value_type;
  using ParentClass = SortedArrayTest< SORTED_ARRAY >;

  using ViewType = std::remove_const_t< typename SORTED_ARRAY::ViewType >;

  /**
   * @brief Test the SortedArrayView copy constructor in regards to memory motion.
   * @param [in] size the number of values to insert.
   */
  void memoryMotionTest( INDEX_TYPE const size )
  {
    ASSERT_TRUE( m_set.empty());

    // Insert values.
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      m_set.insert( T( i ));
    }

    ASSERT_EQ( m_set.size(), size );

    // Capture a view on the device.
    ViewType const view = m_set.toView();
    forall< POLICY >( size, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          LVARRAY_ERROR_IF( view[ i ] != T( i ), "Values changed when moved." );
        } );

    // Change the values.
    m_set.clear();
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      m_set.insert( T( i * i ));
    }

    ASSERT_EQ( m_set.size(), size );

    // Capture the view on host and check that the values haven't been overwritten.
    forall< serialPolicy >( size, [view] ( INDEX_TYPE const i )
    {
      EXPECT_EQ( view[ i ], T( i * i ));
    } );
  }

  /**
   * @brief Test the SortedArray move method.
   * @param [in] size the number of values to insert.
   */
  void memoryMotionMoveTest( INDEX_TYPE const size )
  {
    ASSERT_TRUE( m_set.empty());

    // Insert values.
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      m_set.insert( T( i ));
    }

    ASSERT_EQ( m_set.size(), size );

    T const * const hostPointer = m_set.data();
    m_set.move( RAJAHelper< POLICY >::space, false );
    T const * const devicePointer = m_set.data();
    forall< POLICY >( size, [devicePointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          LVARRAY_ERROR_IF( devicePointer[ i ] != T( i ), "Values changed when moved." );
        } );

    // Change the values.
    for( INDEX_TYPE i = 0; i < size; ++i )
    { const_cast< T & >( hostPointer[ i ] ) = T( 2 * i ); }

    ASSERT_EQ( m_set.size(), size );

    // Move the array back to the host and check that the values haven't been overwritten.
    m_set.move( MemorySpace::host );
    EXPECT_EQ( hostPointer, m_set.data() );
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      EXPECT_EQ( m_set[ i ], T( 2 * i ));
    }
  }

  /**
   * @brief Test the contains method of the SortedArray on device.
   * @param [in] size the number of values to insert.
   */
  void containsDeviceTest( INDEX_TYPE const size )
  {
    ASSERT_TRUE( m_set.empty());

    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      m_set.insert( T( 2 * i ));
    }

    ASSERT_EQ( m_set.size(), size );

    ViewType const view = m_set.toView();
    forall< POLICY >( size, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          LVARRAY_ERROR_IF( !view.contains( T( 2 * i )), "view should contain even numbers." );
          LVARRAY_ERROR_IF( view.contains( T( 2 * i + 1 )), "view should not contain odd numbers." );
        } );
  }

protected:
  using ParentClass::m_set;
  using ParentClass::m_ref;
};

using SortedArrayViewTestTypes = ::testing::Types<
  std::pair< SortedArray< int, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< SortedArray< Tensor, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
  , std::pair< SortedArray< TestString, INDEX_TYPE, DEFAULT_BUFFER >, serialPolicy >
#if ( defined(LVARRAY_USE_CUDA) || defined( LVARRAY_USE_HIP ) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< SortedArray< int, INDEX_TYPE, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< SortedArray< Tensor, INDEX_TYPE, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( SortedArrayViewTest, SortedArrayViewTestTypes, );

TYPED_TEST( SortedArrayViewTest, memoryMotion )
{
  this->memoryMotionTest( 1000 );
}

TYPED_TEST( SortedArrayViewTest, memoryMotionMove )
{
  this->memoryMotionMoveTest( 1000 );
}

TYPED_TEST( SortedArrayViewTest, containsDevice )
{
  this->containsDeviceTest( 1000 );
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
