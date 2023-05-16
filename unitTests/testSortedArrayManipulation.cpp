/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "sortedArrayManipulation.hpp"
#include "testUtils.hpp"
#include "Array.hpp"

// TPL inclues
#include <gtest/gtest.h>

// System includes
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <set>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< typename T >
using Array1D = Array< T, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >;

template< typename T >
using ArrayView1D = ArrayView< T, 1, 0, INDEX_TYPE, DEFAULT_BUFFER >;

template< class T_COMP_POLICY >
class SingleArrayTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_COMP_POLICY >;
  using COMP = std::tuple_element_t< 1, T_COMP_POLICY >;
  using POLICY = std::tuple_element_t< 2, T_COMP_POLICY >;

  void testMakeSorted( INDEX_TYPE const maxSize )
  {
    for( INDEX_TYPE size = 0; size < maxSize; size = INDEX_TYPE( size * 1.5 + 1 ))
    {
      fill( size );

      bool const isInitiallySorted = std::is_sorted( m_ref.begin(), m_ref.end(), m_comp );
      std::sort( m_ref.begin(), m_ref.end(), m_comp );

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > resultOfFirstIsSorted( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > resultOfSecondIsSorted( 0 );
      ArrayView1D< T > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [resultOfFirstIsSorted, resultOfSecondIsSorted, view, comp]
                        LVARRAY_HOST_DEVICE ( INDEX_TYPE )
          {
            resultOfFirstIsSorted += sortedArrayManipulation::isSorted( view.begin(), view.end(), comp );
            sortedArrayManipulation::makeSorted( view.begin(), view.end(), comp );
            resultOfSecondIsSorted += sortedArrayManipulation::isSorted( view.begin(), view.end(), comp );
          } );

      EXPECT_EQ( resultOfFirstIsSorted.get(), isInitiallySorted );
      EXPECT_TRUE( resultOfSecondIsSorted.get() );

      m_array.move( MemorySpace::host );
      checkEquality();
    }
  }

  void testRemoveDuplicates( INDEX_TYPE const maxSize )
  {
    for( INDEX_TYPE size = 1; size < maxSize; size = INDEX_TYPE( size * 1.5 + 1 ) )
    {
      fill( size );

      bool isInitiallySortedUnique;
      {
        std::set< T, COMP > refSet( m_ref.begin(), m_ref.end(), m_comp );
        isInitiallySortedUnique = std::is_sorted( m_ref.begin(), m_ref.end(), m_comp ) &&
                                  ( m_ref.size() == refSet.size() );
        m_ref.clear();
        // workaround bug/feature in gcc12 ( error: iteration 4611686018427387903 invokes undefined behavior )
        //m_ref.insert( m_ref.begin(), refSet.begin(), refSet.end() );
        m_ref.resize( refSet.size());
        std::copy( refSet.begin(), refSet.end(), m_ref.begin() );

        EXPECT_TRUE( sortedArrayManipulation::isSortedUnique( m_ref.begin(), m_ref.end(), m_comp ) );
      }

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > firstIsSortedUnique( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > secondIsSortedUnique( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > numUniqueValues( 0 );
      ArrayView1D< T > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [firstIsSortedUnique, secondIsSortedUnique, numUniqueValues, view, comp]
                        LVARRAY_HOST_DEVICE ( INDEX_TYPE )
          {
            firstIsSortedUnique += sortedArrayManipulation::isSortedUnique( view.begin(), view.end(), comp );
            sortedArrayManipulation::makeSorted( view.begin(), view.end(), comp );

            INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( view.begin(), view.end(), comp );
            numUniqueValues += numUnique;
            secondIsSortedUnique += sortedArrayManipulation::isSortedUnique( view.begin(), view.begin() + numUnique, comp );
          } );

      EXPECT_EQ( firstIsSortedUnique.get(), isInitiallySortedUnique );
      EXPECT_TRUE( secondIsSortedUnique.get() );

      m_array.move( MemorySpace::host );

      EXPECT_EQ( numUniqueValues.get(), m_ref.size() );
      m_array.resize( numUniqueValues.get() );

      checkEquality();
    }
  }

  void testMakeSortedUnique( INDEX_TYPE const maxSize )
  {
    for( INDEX_TYPE size = 1; size < maxSize; size = INDEX_TYPE( size * 1.5 + 1 ) )
    {
      fill( size );

      {
        std::set< T, COMP > refSet( m_ref.begin(), m_ref.end(), m_comp );
        m_ref.clear();
        m_ref.insert( m_ref.begin(), refSet.begin(), refSet.end() );
        EXPECT_TRUE( sortedArrayManipulation::isSortedUnique( m_ref.begin(), m_ref.end(), m_comp ) );
      }

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, INDEX_TYPE > numUniqueValues( 0 );
      ArrayView1D< T > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [numUniqueValues, view, comp] LVARRAY_HOST_DEVICE ( INDEX_TYPE )
          {
            numUniqueValues += sortedArrayManipulation::makeSortedUnique( view.begin(), view.end(), comp );
          } );

      m_array.move( MemorySpace::host );

      EXPECT_EQ( numUniqueValues.get(), m_ref.size() );
      m_array.resize( numUniqueValues.get() );

      checkEquality();
    }
  }

protected:

  void fill( INDEX_TYPE const size )
  {
    std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, 2 * size );

    m_array.resize( size );
    m_ref.resize( size );
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      T const val = T( valueDist( m_gen ) );
      m_array[ i ] = val;
      m_ref[ i ] = val;
    }
  }

  void checkEquality() const
  {
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array[ i ], m_ref[ i ] );
    }
  }

  Array1D< T > m_array;
  std::vector< T > m_ref;
  COMP m_comp;
  std::mt19937_64 m_gen;
};

using SingleArrayTestTypes = ::testing::Types<
  std::tuple< int, sortedArrayManipulation::less< int >, serialPolicy >
  , std::tuple< int, sortedArrayManipulation::greater< int >, serialPolicy >
  , std::tuple< Tensor, sortedArrayManipulation::less< Tensor >, serialPolicy >
  , std::tuple< Tensor, sortedArrayManipulation::greater< Tensor >, serialPolicy >
  , std::tuple< TestString, sortedArrayManipulation::less< TestString >, serialPolicy >
  , std::tuple< TestString, sortedArrayManipulation::greater< TestString >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< int, sortedArrayManipulation::less< int >, parallelDevicePolicy< 256 > >
  , std::tuple< int, sortedArrayManipulation::greater< int >, parallelDevicePolicy< 256 > >
  , std::tuple< Tensor, sortedArrayManipulation::less< Tensor >, parallelDevicePolicy< 256 > >
  , std::tuple< Tensor, sortedArrayManipulation::greater< Tensor >, parallelDevicePolicy< 256 > >
#endif
  >;
TYPED_TEST_SUITE( SingleArrayTest, SingleArrayTestTypes, );

TYPED_TEST( SingleArrayTest, makeSorted )
{
  this->testMakeSorted( 500 );
}

TYPED_TEST( SingleArrayTest, removeDuplicates )
{
  this->testRemoveDuplicates( 500 );
}

TYPED_TEST( SingleArrayTest, makeSortedUnique )
{
  this->testMakeSortedUnique( 500 );
}

template< class KEY_T_COMP_POLICY >
class DualArrayTest : public ::testing::Test
{
public:
  using KEY = std::tuple_element_t< 0, KEY_T_COMP_POLICY >;
  using T = std::tuple_element_t< 1, KEY_T_COMP_POLICY >;
  using COMP = std::tuple_element_t< 2, KEY_T_COMP_POLICY >;
  using POLICY = std::tuple_element_t< 3, KEY_T_COMP_POLICY >;

  void testMakeSorted( INDEX_TYPE maxSize )
  {
    for( INDEX_TYPE size = 0; size < maxSize; size = INDEX_TYPE( size * 1.5 + 1 ))
    {
      fillArrays( size );
      std::sort( m_ref.begin(), m_ref.end(), PairComp< KEY, T, COMP >());

      ArrayView1D< KEY > const & keys = m_keys;
      ArrayView1D< T > const & values = m_values;
      COMP & comp = m_comp;
      forall< POLICY >( 1, [keys, values, comp] LVARRAY_HOST_DEVICE ( INDEX_TYPE )
          {
            dualSort( keys.begin(), keys.end(), values.begin(), comp );
          } );

      m_keys.move( MemorySpace::host );
      m_values.move( MemorySpace::host );
      checkEquality();
    }
  }

private:

  void fillArrays( INDEX_TYPE const size )
  {
    std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, 2 * size );

    m_keys.resize( size );
    m_values.resize( size );
    m_ref.resize( size );
    for( INDEX_TYPE i = 0; i < size; ++i )
    {
      INDEX_TYPE const seed = valueDist( m_gen );
      KEY const key = KEY( seed );
      T const val = T( seed * ( seed - size ) );
      m_keys[ i ] = key;
      m_values[ i ] = val;
      m_ref[ i ] = std::pair< KEY, T >( key, val );
    }
  }

  void checkEquality()
  {
    for( INDEX_TYPE i = 0; i < m_keys.size(); ++i )
    {
      EXPECT_EQ( m_keys[ i ], m_ref[ i ].first );
      EXPECT_EQ( m_values[ i ], m_ref[ i ].second );
    }
  }

  Array1D< KEY > m_keys;
  Array1D< T > m_values;
  std::vector< std::pair< KEY, T > > m_ref;
  COMP m_comp;
  std::mt19937_64 m_gen;
};

using DualArrayTestTypes = ::testing::Types<
  std::tuple< int, int, sortedArrayManipulation::less< int >, serialPolicy >
  , std::tuple< int, int, sortedArrayManipulation::greater< int >, serialPolicy >
  , std::tuple< Tensor, Tensor, sortedArrayManipulation::less< Tensor >, serialPolicy >
  , std::tuple< Tensor, Tensor, sortedArrayManipulation::greater< Tensor >, serialPolicy >
  , std::tuple< int, Tensor, sortedArrayManipulation::less< int >, serialPolicy >
  , std::tuple< Tensor, int, sortedArrayManipulation::greater< Tensor >, serialPolicy >
  , std::tuple< TestString, TestString, sortedArrayManipulation::less< TestString >, serialPolicy >
  , std::tuple< TestString, TestString, sortedArrayManipulation::greater< TestString >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) )&& defined(LVARRAY_USE_CHAI)
  , std::tuple< int, int, sortedArrayManipulation::less< int >, parallelDevicePolicy< 256 > >
  , std::tuple< int, int, sortedArrayManipulation::greater< int >, parallelDevicePolicy< 256 > >
  , std::tuple< Tensor, Tensor, sortedArrayManipulation::less< Tensor >, parallelDevicePolicy< 256 > >
  , std::tuple< Tensor, Tensor, sortedArrayManipulation::greater< Tensor >, parallelDevicePolicy< 256 > >
  , std::tuple< int, Tensor, sortedArrayManipulation::less< int >, parallelDevicePolicy< 256 > >
  , std::tuple< Tensor, int, sortedArrayManipulation::greater< Tensor >, parallelDevicePolicy< 256 > >
#endif
  >;
TYPED_TEST_SUITE( DualArrayTest, DualArrayTestTypes, );

TYPED_TEST( DualArrayTest, makeSorted )
{
  this->testMakeSorted( 250 );
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
