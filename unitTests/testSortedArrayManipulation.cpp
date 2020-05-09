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

template< class T_COMP_POLICY >
class SingleArrayTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_COMP_POLICY >;
  using COMP = std::tuple_element_t< 1, T_COMP_POLICY >;
  using POLICY = std::tuple_element_t< 2, T_COMP_POLICY >;

  void testMakeSorted( std::ptrdiff_t const maxSize )
  {
    for( std::ptrdiff_t size = 0; size < maxSize; size = std::ptrdiff_t( size * 1.5 + 1 ))
    {
      fill( size );

      bool const isInitiallySorted = std::is_sorted( m_ref.begin(), m_ref.end(), m_comp );
      std::sort( m_ref.begin(), m_ref.end(), m_comp );

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > resultOfFirstIsSorted( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > resultOfSecondIsSorted( 0 );
      ArrayView< T, 1 > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [resultOfFirstIsSorted, resultOfSecondIsSorted, view, comp]
                        LVARRAY_HOST_DEVICE ( std::ptrdiff_t )
          {
            resultOfFirstIsSorted += sortedArrayManipulation::isSorted( view.begin(), view.end(), comp );
            sortedArrayManipulation::makeSorted( view.begin(), view.end(), comp );
            resultOfSecondIsSorted += sortedArrayManipulation::isSorted( view.begin(), view.end(), comp );
          } );

      EXPECT_EQ( resultOfFirstIsSorted.get(), isInitiallySorted );
      EXPECT_TRUE( resultOfSecondIsSorted.get() );

      m_array.move( chai::CPU );
      checkEquality();
    }
  }

  void testRemoveDuplicates( std::ptrdiff_t const maxSize )
  {
    for( std::ptrdiff_t size = 1; size < maxSize; size = std::ptrdiff_t( size * 1.5 + 1 ) )
    {
      fill( size );

      bool isInitiallySortedUnique;
      {
        std::set< T, COMP > refSet( m_ref.begin(), m_ref.end(), m_comp );
        isInitiallySortedUnique = std::is_sorted( m_ref.begin(), m_ref.end(), m_comp ) &&
                                  ( m_ref.size() == refSet.size() );
        m_ref.clear();
        m_ref.insert( m_ref.begin(), refSet.begin(), refSet.end() );
        EXPECT_TRUE( sortedArrayManipulation::isSortedUnique( m_ref.begin(), m_ref.end(), m_comp ) );
      }

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > firstIsSortedUnique( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > secondIsSortedUnique( 0 );
      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > numUniqueValues( 0 );
      ArrayView< T, 1 > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [firstIsSortedUnique, secondIsSortedUnique, numUniqueValues, view, comp]
                        LVARRAY_HOST_DEVICE ( std::ptrdiff_t )
          {
            firstIsSortedUnique += sortedArrayManipulation::isSortedUnique( view.begin(), view.end(), comp );
            sortedArrayManipulation::makeSorted( view.begin(), view.end(), comp );

            std::ptrdiff_t const numUnique = sortedArrayManipulation::makeSortedUnique( view.begin(), view.end(), comp );
            numUniqueValues += numUnique;
            secondIsSortedUnique += sortedArrayManipulation::isSortedUnique( view.begin(), view.begin() + numUnique, comp );
          } );

      EXPECT_EQ( firstIsSortedUnique.get(), isInitiallySortedUnique );
      EXPECT_TRUE( secondIsSortedUnique.get() );

      m_array.move( chai::CPU );

      EXPECT_EQ( numUniqueValues.get(), m_ref.size() );
      m_array.resize( numUniqueValues.get() );

      checkEquality();
    }
  }

  void testMakeSortedUnique( std::ptrdiff_t const maxSize )
  {
    for( std::ptrdiff_t size = 1; size < maxSize; size = std::ptrdiff_t( size * 1.5 + 1 ) )
    {
      fill( size );

      {
        std::set< T, COMP > refSet( m_ref.begin(), m_ref.end(), m_comp );
        m_ref.clear();
        m_ref.insert( m_ref.begin(), refSet.begin(), refSet.end() );
        EXPECT_TRUE( sortedArrayManipulation::isSortedUnique( m_ref.begin(), m_ref.end(), m_comp ) );
      }

      RAJA::ReduceSum< typename RAJAHelper< POLICY >::ReducePolicy, std::ptrdiff_t > numUniqueValues( 0 );
      ArrayView< T, 1 > const & view = m_array;
      COMP & comp = m_comp;

      forall< POLICY >( 1, [numUniqueValues, view, comp] LVARRAY_HOST_DEVICE ( std::ptrdiff_t )
          {
            numUniqueValues += sortedArrayManipulation::makeSortedUnique( view.begin(), view.end(), comp );
          } );

      m_array.move( chai::CPU );

      EXPECT_EQ( numUniqueValues.get(), m_ref.size() );
      m_array.resize( numUniqueValues.get() );

      checkEquality();
    }
  }

protected:

  void fill( std::ptrdiff_t const size )
  {
    std::uniform_int_distribution< std::ptrdiff_t > valueDist( 0, 2 * size );

    m_array.resize( size );
    m_ref.resize( size );
    for( std::ptrdiff_t i = 0; i < size; ++i )
    {
      T const val = T( valueDist( m_gen ) );
      m_array[ i ] = val;
      m_ref[ i ] = val;
    }
  }

  void checkEquality() const
  {
    for( std::ptrdiff_t i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array[ i ], m_ref[ i ] );
    }
  }

  Array< T, 1 > m_array;
  std::vector< T > m_ref;
  COMP m_comp;
  std::mt19937_64 m_gen;
};

using SingleArrayTestTypes = ::testing::Types<
  std::tuple< int, sortedArrayManipulation::less< int >, serialPolicy >
  , std::tuple< int, sortedArrayManipulation::greater< int >, serialPolicy >
  , std::tuple< Tensor, sortedArrayManipulation::greater< Tensor >, serialPolicy >
  , std::tuple< Tensor, sortedArrayManipulation::greater< Tensor >, serialPolicy >
  , std::tuple< TestString, sortedArrayManipulation::greater< TestString >, serialPolicy >
  , std::tuple< TestString, sortedArrayManipulation::greater< TestString >, serialPolicy >

#ifdef USE_CUDA
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

  void testMakeSorted( std::ptrdiff_t maxSize )
  {
    for( std::ptrdiff_t size = 0; size < maxSize; size = std::ptrdiff_t( size * 1.5 + 1 ))
    {
      fillArrays( size );
      std::sort( m_ref.begin(), m_ref.end(), PairComp< KEY, T, COMP >());

      ArrayView< KEY, 1 > const & keys = m_keys;
      ArrayView< T, 1 > const & values = m_values;
      COMP & comp = m_comp;
      forall< POLICY >( 1, [keys, values, comp] LVARRAY_HOST_DEVICE ( std::ptrdiff_t )
          {
            dualSort( keys.begin(), keys.end(), values.begin(), comp );
          } );

      m_keys.move( chai::CPU );
      m_values.move( chai::CPU );
      checkEquality();
    }
  }

private:

  void fillArrays( std::ptrdiff_t const size )
  {
    std::uniform_int_distribution< std::ptrdiff_t > valueDist( 0, 2 * size );

    m_keys.resize( size );
    m_values.resize( size );
    m_ref.resize( size );
    for( std::ptrdiff_t i = 0; i < size; ++i )
    {
      std::ptrdiff_t const seed = valueDist( m_gen );
      KEY const key = KEY( seed );
      T const val = T( seed * ( seed - size ) );
      m_keys[ i ] = key;
      m_values[ i ] = val;
      m_ref[ i ] = std::pair< KEY, T >( key, val );
    }
  }

  void checkEquality()
  {
    for( std::ptrdiff_t i = 0; i < m_keys.size(); ++i )
    {
      EXPECT_EQ( m_keys[ i ], m_ref[ i ].first );
      EXPECT_EQ( m_values[ i ], m_ref[ i ].second );
    }
  }

  Array< KEY, 1 > m_keys;
  Array< T, 1 > m_values;
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

#ifdef USE_CUDA
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
