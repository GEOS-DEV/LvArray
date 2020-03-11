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

#if defined(USE_CUDA) && defined(USE_CHAI)
  #include <chai/util/forall.hpp>
#endif

// System includes
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <set>

namespace LvArray
{
namespace sortedArrayManipulation
{
namespace testing
{

using namespace LvArray::testing;

using INDEX_TYPE = std::ptrdiff_t;

template< class T >
using Array1d = LvArray::Array< T, 1 >;

template< class T >
using ArrayView1d = LvArray::ArrayView< T, 1 > const;


// A static random number generator is used so that subsequent calls to the same test method will
// do different things. This initializes the generator with the default seed so the test behavior will
// be uniform across runs and platforms. However the behavior of an individual method will depend on the state
// of the generator and hence on the calls before it.
static std::mt19937_64 gen;

template< class T >
void checkEqual( Array1d< T > const & a, std::vector< T > const & b )
{
  for( INDEX_TYPE i = 0; i < a.size(); ++i )
  {
    EXPECT_EQ( a[i], b[i] );
  }
}

template< class SORTINGTYPE, class T >
void checkEqual( Array1d< SORTINGTYPE > const & a, Array1d< T > const & b, std::vector< std::pair< T, SORTINGTYPE > > const & c )
{
  for( INDEX_TYPE i = 0; i < a.size(); ++i )
  {
    EXPECT_EQ( a[i], c[i].second );
    EXPECT_EQ( b[i], c[i].first );
  }
}

template< class T, class Compare >
void check( Array1d< T > & a, std::vector< T > & b, Compare comp )
{
  makeSorted( a.begin(), a.end(), comp );
  std::sort( b.begin(), b.end(), comp );
  checkEqual( a, b );
}

template< class SORTINGTYPE, class T, class Compare >
void check( Array1d< SORTINGTYPE > & a, Array1d< T > & b, std::vector< std::pair< T, SORTINGTYPE > > & c, Compare comp )
{
  dualSort( a.begin(), a.end(), b.begin(), comp );
  std::sort( c.begin(), c.end(), PairComp< T, SORTINGTYPE, Compare >());
  checkEqual( a, b, c );
}

template< class T >
void fillArrays( INDEX_TYPE size, Array1d< T > & a, std::vector< T > & b )
{
  std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, size );

  a.resize( size );
  b.resize( size );
  for( INDEX_TYPE i = 0; i < size; ++i )
  {
    T const val = T( valueDist( gen ));
    a[i] = val;
    b[i] = val;
  }
}

template< class SORTINGTYPE, class T >
void fillArrays( INDEX_TYPE size, Array1d< SORTINGTYPE > & a, Array1d< T > & b, std::vector< std::pair< T, SORTINGTYPE > > & c )
{
  std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, size );

  a.resize( size );
  b.resize( size );
  c.resize( size );
  for( INDEX_TYPE i = 0; i < size; ++i )
  {
    INDEX_TYPE const seed = valueDist( gen );
    SORTINGTYPE const sortingVal = SORTINGTYPE( seed );
    T const val = T( seed * (seed + 1 - size));
    a[i] = sortingVal;
    b[i] = val;
    c[i] = std::pair< T, SORTINGTYPE >( val, sortingVal );
  }
}


template< class T, class Compare >
void correctnessTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 2000;
  Array1d< T > a( MAX_SIZE );
  std::vector< T > b( MAX_SIZE );

  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    fillArrays( size, a, b );
    check( a, b, Compare());
  }
}

template< class SORTINGTYPE, class T, class Compare >
void dualCorrectnessTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 2000;
  Array1d< SORTINGTYPE > a( MAX_SIZE );
  Array1d< T > b( MAX_SIZE );
  std::vector< std::pair< T, SORTINGTYPE > > c( MAX_SIZE );

  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    fillArrays( size, a, b, c );
    check( a, b, c, Compare());
  }
}

template< class T, class Compare >
void performanceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 1024 * 1024 * 8;

  std::cout << std::setiosflags( std::ios::fixed ) << std::setprecision( 5 );

  Array1d< T > a( MAX_SIZE );
  std::vector< T > b( MAX_SIZE );

  for( INDEX_TYPE size = 0; size <= MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    std::chrono::duration< double > sortingTime {};
    std::chrono::duration< double > stdTime {};
    for( INDEX_TYPE iter = 0; iter < MAX_SIZE / (size + 1) + 1; ++iter )
    {
      fillArrays( size, a, b );

      auto start = std::chrono::high_resolution_clock::now();
      makeSorted( a.begin(), a.end(), Compare());
      auto end = std::chrono::high_resolution_clock::now();
      sortingTime += end - start;

      start = std::chrono::high_resolution_clock::now();
      std::sort( b.begin(), b.end(), Compare());
      end = std::chrono::high_resolution_clock::now();
      stdTime += end - start;

      fillArrays( size, a, b );

      start = std::chrono::high_resolution_clock::now();
      std::sort( b.begin(), b.end(), Compare());
      end = std::chrono::high_resolution_clock::now();
      stdTime += end - start;

      start = std::chrono::high_resolution_clock::now();
      makeSorted( a.begin(), a.end(), Compare());
      end = std::chrono::high_resolution_clock::now();
      sortingTime += end - start;

      checkEqual( a, b );
    }

    if( sortingTime.count() > stdTime.count())
    {
      std::cout << "size = " << size << ",\tmakeSorted : " << sortingTime.count() <<
        ", std::sort : " << stdTime.count() << ", slower by : " << (sortingTime.count() / stdTime.count()) * 100.0 - 100.0 << "%" << std::endl;
    }
    else
    {
      std::cout << "size = " << size << ",\tmakeSorted : " << sortingTime.count() <<
        ", std::sort : " << stdTime.count() << ", FASTER!!!" << std::endl;
    }
  }
}

template< class SORTINGTYPE, class T, class Compare >
void dualPerformanceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 1024 * 1024 * 8;

  std::cout << std::setiosflags( std::ios::fixed ) << std::setprecision( 5 );

  Array1d< SORTINGTYPE > a( MAX_SIZE );
  Array1d< T > b( MAX_SIZE );
  std::vector< std::pair< T, SORTINGTYPE > > c( MAX_SIZE );

  for( INDEX_TYPE size = 0; size <= MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    std::chrono::duration< double > sortingTime {};
    std::chrono::duration< double > stdTime {};
    for( INDEX_TYPE iter = 0; iter < MAX_SIZE / (size + 1) + 1; ++iter )
    {
      fillArrays( size, a, b, c );

      auto start = std::chrono::high_resolution_clock::now();
      dualSort( a.begin(), a.end(), b.begin(), Compare());
      auto end = std::chrono::high_resolution_clock::now();
      sortingTime += end - start;

      start = std::chrono::high_resolution_clock::now();
      std::sort( c.begin(), c.end(), PairComp< T, SORTINGTYPE, Compare >());
      end = std::chrono::high_resolution_clock::now();
      stdTime += end - start;

      fillArrays( size, a, b, c );

      start = std::chrono::high_resolution_clock::now();
      std::sort( c.begin(), c.end(), PairComp< T, SORTINGTYPE, Compare >());
      end = std::chrono::high_resolution_clock::now();
      stdTime += end - start;

      start = std::chrono::high_resolution_clock::now();
      dualSort( a.begin(), a.end(), b.begin(), Compare());
      end = std::chrono::high_resolution_clock::now();
      sortingTime += end - start;

      checkEqual( a, b, c );
    }

    if( sortingTime.count() > stdTime.count())
    {
      std::cout << "size = " << size << ",\tmakeSorted : " << sortingTime.count() <<
        ", std::sort : " << stdTime.count() << ", slower by : " << (sortingTime.count() / stdTime.count()) * 100.0 - 100.0 << "%" << std::endl;
    }
    else
    {
      std::cout << "size = " << size << ",\tmakeSorted : " << sortingTime.count() <<
        ", std::sort : " << stdTime.count() << ", FASTER!!!" << std::endl;
    }
  }
}

template< class T >
void removeDuplicatesCorrectnessTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 2000;
  std::vector< T > values( MAX_SIZE );

  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    values.resize( size );

    std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, size * 2 );

    for( T & val : values )
    {
      val = T( valueDist( gen ));
    }

    std::set< T > ref( values.begin(), values.end());

    std::sort( values.begin(), values.end());
    ASSERT_TRUE( sortedArrayManipulation::isSorted( values.data(), values.size()));

    INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( values.data(), values.size());
    EXPECT_EQ( numUniqueValues, ref.size());

    EXPECT_TRUE( sortedArrayManipulation::allUnique( values.data(), numUniqueValues ));

    INDEX_TYPE i = 0;
    for( T const & refVal : ref )
    {
      EXPECT_EQ( refVal, values[i] );
      ++i;
    }
  }
}

#ifdef USE_CUDA

template< class T, class Compare >
void correctnessDeviceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 100;
  Array1d< T > a( MAX_SIZE );
  std::vector< T > b( MAX_SIZE );
  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    fillArrays( size, a, b );

    ArrayView1d< T > & aView = a;
    forall( gpu(), 0, 1,
            [=] __device__ ( INDEX_TYPE i )
          {
            makeSorted( aView.begin(), aView.end(), Compare());
          }
            );

    a.move( chai::CPU );

    std::sort( b.begin(), b.end(), Compare());
    checkEqual( a, b );
  }
}

template< class SORTINGTYPE, class T, class Compare >
void dualCorrectnessDeviceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 100;
  Array1d< SORTINGTYPE > a( MAX_SIZE );
  Array1d< T > b( MAX_SIZE );
  std::vector< std::pair< T, SORTINGTYPE > > c( MAX_SIZE );

  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    fillArrays( size, a, b, c );

    ArrayView1d< SORTINGTYPE > & aView = a;
    ArrayView1d< T > & bView = b;
    forall( gpu(), 0, 1,
            [=] __device__ ( INDEX_TYPE i )
          {
            dualSort( aView.begin(), aView.end(), bView.begin(), Compare());
          }
            );

    a.move( chai::CPU );
    b.move( chai::CPU );

    std::sort( c.begin(), c.end(), PairComp< T, SORTINGTYPE, Compare >());
    checkEqual( a, b, c );
  }
}

template< class T >
void removeDuplicatesCorrectnessDeviceTest()
{
  constexpr INDEX_TYPE MAX_SIZE = 2000;
  Array1d< T > values( MAX_SIZE );
  Array1d< INDEX_TYPE > numUniqueValues( 1 );

  for( INDEX_TYPE size = 0; size < MAX_SIZE; size = INDEX_TYPE( size * 1.5 + 1 ))
  {
    values.resize( size );

    std::uniform_int_distribution< INDEX_TYPE > valueDist( 0, size * 2 );

    for( T & val : values )
    {
      val = T( valueDist( gen ));
    }

    std::set< T > ref( values.begin(), values.end());

    std::sort( values.begin(), values.end());
    ASSERT_TRUE( sortedArrayManipulation::isSorted( values.data(), values.size()));

    forall( gpu(), 0, 1,
            [valuesView=values.toView(), numUniqueValuesView=numUniqueValues.toView()] __device__ ( INDEX_TYPE i )
          {
            numUniqueValuesView[0] = sortedArrayManipulation::removeDuplicates( valuesView.data(), valuesView.size());
            LVARRAY_ERROR_IF( !sortedArrayManipulation::allUnique( valuesView.data(), numUniqueValuesView[0] ), "Values should be unique!" );
          }
            );

    values.move( chai::CPU );
    numUniqueValues.move( chai::CPU );
    EXPECT_EQ( numUniqueValues[0], ref.size());

    INDEX_TYPE i = 0;
    for( T const & refVal : ref )
    {
      EXPECT_EQ( refVal, values[i] );
      ++i;
    }
  }
}

#endif

TEST( sort, correctness )
{
  correctnessTest< int, less< int > >();
  correctnessTest< int, greater< int > >();

  correctnessTest< Tensor, less< Tensor > >();
  correctnessTest< Tensor, greater< Tensor > >();

  correctnessTest< TestString, less< TestString > >();
  correctnessTest< TestString, greater< TestString > >();
}

TEST( dualSort, correctness )
{
  dualCorrectnessTest< int, int, less< int > >();
  dualCorrectnessTest< int, Tensor, greater< int > >();

  dualCorrectnessTest< Tensor, int, less< Tensor > >();
  dualCorrectnessTest< Tensor, TestString, greater< Tensor > >();

  dualCorrectnessTest< TestString, int, less< TestString > >();
  dualCorrectnessTest< TestString, Tensor, greater< TestString > >();
}

TEST( sort, performance )
{
  // performanceTest<int, less<int>>();
  SUCCEED();
}

TEST( dualSort, performance )
{
  // dualPerformanceTest<int, int, less<int>>();
  SUCCEED();
}

TEST( removeDuplicates, correctness )
{
  removeDuplicatesCorrectnessTest< int >();
  removeDuplicatesCorrectnessTest< Tensor >();
  removeDuplicatesCorrectnessTest< TestString >();
}

#ifdef USE_CUDA

CUDA_TEST( sort, correctnessDevice )
{
  correctnessDeviceTest< int, less< int > >();
  correctnessDeviceTest< int, greater< int > >();

  correctnessDeviceTest< Tensor, less< Tensor > >();
  correctnessDeviceTest< Tensor, greater< Tensor > >();
}

CUDA_TEST( dualSort, correctnessDevice )
{
  dualCorrectnessDeviceTest< int, int, less< int > >();
  dualCorrectnessDeviceTest< int, Tensor, greater< int > >();

  dualCorrectnessDeviceTest< Tensor, int, less< Tensor > >();
  dualCorrectnessDeviceTest< Tensor, Tensor, greater< Tensor > >();
}

CUDA_TEST( removeDuplicates, correctnessDevice )
{
  removeDuplicatesCorrectnessDeviceTest< int >();
  removeDuplicatesCorrectnessDeviceTest< Tensor >();
}


#endif

} // namespace testing
} // namespace sortedArrayManipulation
} // namespace LvArray
