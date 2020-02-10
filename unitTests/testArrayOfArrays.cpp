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
#include "ArrayOfArrays.hpp"
#include "testUtils.hpp"
#include "Array.hpp"

// TPL includes
#include <gtest/gtest.h>

#ifdef USE_CHAI
  #include <chai/util/forall.hpp>
#endif

// System includes
#include <vector>
#include <random>

#ifdef USE_OMP
  #include <omp.h>
#endif


namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< class T, bool CONST_SIZES=false >
using ViewType = ArrayOfArraysView< T, INDEX_TYPE const, CONST_SIZES >;

template< class T >
void compareToReference( ViewType< T const, true > const & view, std::vector< std::vector< T > > const & ref )
{
  ASSERT_EQ( view.size(), ref.size());

  for( INDEX_TYPE i = 0 ; i < view.size() ; ++i )
  {
    ASSERT_EQ( view.sizeOfArray( i ), ref[i].size());

    for( INDEX_TYPE j = 0 ; j < view.sizeOfArray( i ) ; ++j )
    {
      EXPECT_EQ( view[i][j], ref[i][j] );
      EXPECT_EQ( view( i, j ), view[i][j] );
    }

    INDEX_TYPE j = 0;
    for( T const & val : view.getIterableArray( i ))
    {
      EXPECT_EQ( val, ref[i][j] );
      ++j;
    }

    EXPECT_EQ( j, view.sizeOfArray( i ));
  }
}

#define COMPARE_TO_REFERENCE( view, ref ) { SCOPED_TRACE( "" ); compareToReference( view, ref ); \
}

template< class T >
void printArray( ViewType< T const, true > const & view )
{
  std::cout << "{" << std::endl;

  for( INDEX_TYPE i = 0 ; i < view.size() ; ++i )
  {
    std::cout << "\t{";
    for( INDEX_TYPE j = 0 ; j < view.sizeOfArray( i ) ; ++j )
    {
      std::cout << view[i][j] << ", ";
    }

    for( INDEX_TYPE j = view.sizeOfArray( i ) ; j < view.capacityOfArray( i ) ; ++j )
    {
      std::cout << "X" << ", ";
    }

    std::cout << "}" << std::endl;
  }

  std::cout << "}" << std::endl;
}

template< class T >
class ArrayOfArraysTest : public ::testing::Test
{
public:

  void appendArray( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    std::vector< T > arrayToAppend( maxAppendSize );

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      arrayToAppend.resize( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        arrayToAppend[j] = T( i * LARGE_NUMBER + j );
      }

      m_array.appendArray( arrayToAppend.data(), arrayToAppend.size());
      m_ref.push_back( arrayToAppend );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void appendArray2( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    std::vector< T > arrayToAppend( maxAppendSize );

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      arrayToAppend.resize( nValues );
      m_array.appendArray( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        arrayToAppend[j] = value;
        m_array( m_array.size() - 1, j ) = value;
      }

      m_ref.push_back( arrayToAppend );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void insertArray( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    std::vector< T > arrayToAppend( maxAppendSize );

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      arrayToAppend.resize( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        arrayToAppend[j] = T( i * LARGE_NUMBER + j );
      }

      INDEX_TYPE insertPos = rand( 0, m_array.size());
      m_array.insertArray( insertPos, arrayToAppend.data(), arrayToAppend.size());
      m_ref.insert( m_ref.begin() + insertPos, arrayToAppend );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void eraseArray( INDEX_TYPE const nArrays )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const removePos = rand( 0, m_array.size() -1 );
      m_array.eraseArray( removePos );
      m_ref.erase( m_ref.begin() + removePos );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void resize( INDEX_TYPE const newSize, INDEX_TYPE const capacityPerArray=0 )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    m_array.resize( newSize, capacityPerArray );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void resize()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const newSize = rand( 0, 2 * m_array.size());
    m_array.resize( newSize );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void resizeArray( INDEX_TYPE const maxSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const newSize = rand( 0, maxSize );

      T const defaultValue = T( i );
      m_array.resizeArray( i, newSize, defaultValue );
      m_ref[i].resize( newSize, defaultValue );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void appendToArray( INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        m_array.appendToArray( i, value );
        m_ref[i].push_back( value );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void atomicAppendToArraySerial( INDEX_TYPE const numAppends )
  {
    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), 0 );
    }

    // Append sequential values to each array
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      for( INDEX_TYPE j = 0 ; j < numAppends ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        m_array.atomicAppendToArray( RAJA::auto_atomic{}, i, value );
      }
    }

    // Now check that the values are as expected.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), numAppends );
      ASSERT_LE( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      for( INDEX_TYPE j = 0 ; j < numAppends ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        EXPECT_EQ( value, m_array( i, j )) << i << ", " << j;
      }
    }
  }

#ifdef USE_OPENMP
  void atomicAppendToArrayOMP( INDEX_TYPE const appendsPerArrayPerThread )
  {
    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), 0 );
    }

    INDEX_TYPE numThreads;

    #pragma omp parallel
    {
      #pragma omp master
      numThreads = omp_get_num_threads();

      INDEX_TYPE const threadNum = omp_get_thread_num();

      // In parallel append sequential values to each array.
      for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
      {
        for( INDEX_TYPE j = 0 ; j < appendsPerArrayPerThread ; ++j )
        {
          T const value = T( i * LARGE_NUMBER + threadNum * appendsPerArrayPerThread + j );
          m_array.atomicAppendToArray( RAJA::auto_atomic{}, i, value );
        }
      }
    }

    LVARRAY_LOG( "Used " << numThreads << " threads." );

    // Now sort each array and check that the values are as expected.
    INDEX_TYPE const appendsPerArray = numThreads * appendsPerArrayPerThread;
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), appendsPerArray );
      ASSERT_LE( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T * const subArrayPtr = m_array[i];
      std::sort( subArrayPtr, subArrayPtr + appendsPerArray );

      for( INDEX_TYPE j = 0 ; j < appendsPerArray ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        EXPECT_EQ( value, m_array( i, j )) << i << ", " << j;
      }
    }
  }
#endif

  void appendMultipleToArray( INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    std::vector< T > valuesToAppend( maxAppendSize );
    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      valuesToAppend.resize( nValues );
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        valuesToAppend[j] = T( i * LARGE_NUMBER + arraySize + j );
      }

      m_array.appendToArray( i, valuesToAppend.data(), valuesToAppend.size());
      m_ref[i].insert( m_ref[i].end(), valuesToAppend.begin(), valuesToAppend.end());
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void insertIntoArray( INDEX_TYPE const maxInsertSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nValues = rand( 0, maxInsertSize );
      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );

        INDEX_TYPE const insertPos = rand( 0, arraySize + j );
        m_array.insertIntoArray( i, insertPos, value );
        m_ref[i].insert( m_ref[i].begin() + insertPos, value );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void insertMultipleIntoArray( INDEX_TYPE const maxInsertSize )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    std::vector< T > valuesToAppend( maxInsertSize );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInsertSize );
      valuesToAppend.resize( nValues );

      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        valuesToAppend[j] = T( i * LARGE_NUMBER + arraySize + j );
      }

      INDEX_TYPE const insertPos = rand( 0, arraySize );

      m_array.insertIntoArray( i, insertPos, valuesToAppend.data(), valuesToAppend.size());
      m_ref[i].insert( m_ref[i].begin() + insertPos, valuesToAppend.begin(), valuesToAppend.end());
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void eraseFromArray( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nToRemove = rand( 0, min( arraySize, maxRemoves ));
      for( INDEX_TYPE j = 0 ; j < nToRemove ; ++j )
      {
        INDEX_TYPE const removePosition = rand( 0, arraySize - j - 1 );

        m_array.eraseFromArray( i, removePosition );
        m_ref[i].erase( m_ref[i].begin() + removePosition );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void removeMultipleFromArray( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      INDEX_TYPE const removePosition = rand( 0, arraySize - 1 );
      INDEX_TYPE const nToRemove = rand( 0, min( maxRemoves, arraySize - removePosition ));

      m_array.eraseFromArray( i, removePosition, nToRemove );
      m_ref[i].erase( m_ref[i].begin() + removePosition, m_ref[i].begin() + removePosition + nToRemove );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void compress()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    m_array.compress();
    T const * const values = m_array[0];

    INDEX_TYPE curOffset = 0;
    for( INDEX_TYPE i = 0 ; i < m_array.size() ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T const * const curValues = m_array[i];
      ASSERT_EQ( values + curOffset, curValues );

      curOffset += m_array.sizeOfArray( i );
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void fill()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    T const * const values = m_array[0];
    T const * const endValues = m_array[m_array.size() - 1];

    for( INDEX_TYPE i = 0 ; i < m_array.size() ; ++i )
    {
      fillArray( i );
    }

    T const * const newValues = m_array[0];
    T const * const newEndValues = m_array[m_array.size() - 1];
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void capacityOfArray( INDEX_TYPE const maxCapacity )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    for( INDEX_TYPE i = 0 ; i < m_array.size() ; ++i )
    {
      INDEX_TYPE const capacity = m_array.capacityOfArray( i );
      INDEX_TYPE const newCapacity = rand( 0, maxCapacity );

      if( newCapacity < capacity )
      {
        T const * const values = m_array[i];
        m_array.setCapacityOfArray( i, newCapacity );
        T const * const newValues = m_array[i];

        ASSERT_EQ( values, newValues );
        m_ref[i].resize( newCapacity );
      }
      else
      {
        m_array.setCapacityOfArray( i, newCapacity );
        fillArray( i );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void deepCopy()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    ArrayOfArrays< T > copy( m_array );

    ASSERT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0 ; i < m_array.size() ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));

      T const * const values = m_array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_NE( values, valuesCopy );

      for( INDEX_TYPE j = 0 ; j < arraySize ; ++j )
      {
        ASSERT_EQ( m_array[i][j], copy[i][j] );
        copy[i][j] = T( -i * LARGE_NUMBER - j );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void shallowCopy()
  {
    ViewType< T > const copy( m_array.toView());

    ASSERT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0 ; i < m_array.size() ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));
      T const * const values = m_array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_EQ( values, valuesCopy );

      for( INDEX_TYPE j = 0 ; j < arraySize ; ++j )
      {
        ASSERT_EQ( m_array[i][j], copy[i][j] );

        copy[i][j] = T( -i * LARGE_NUMBER - j );

        ASSERT_EQ( m_array[i][j], T( -i * LARGE_NUMBER - j ));
        ASSERT_EQ( m_array[i][j], copy[i][j] );
      }
    }
  }

protected:

  void fillArray( INDEX_TYPE const i )
  {
    INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
    INDEX_TYPE const nToAppend = m_array.capacityOfArray( i ) - arraySize;
    T const * const values = m_array[i];
    T const * const endValues = m_array[m_array.size() - 1];
    INDEX_TYPE const capacity = m_array.capacityOfArray( i );

    for( INDEX_TYPE j = 0 ; j < nToAppend ; ++j )
    {
      T const valueToAppend( LARGE_NUMBER * i + j );
      m_array.appendToArray( i, valueToAppend );
      m_ref[i].push_back( valueToAppend );
    }

    T const * const newValues = m_array[i];
    T const * const newEndValues = m_array[m_array.size() - 1];
    INDEX_TYPE const endCapacity = m_array.capacityOfArray( i );
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );
    ASSERT_EQ( capacity, endCapacity );
  }

  INDEX_TYPE rand( INDEX_TYPE const min, INDEX_TYPE const max )
  {
    return std::uniform_int_distribution< INDEX_TYPE >( min, max )( m_gen );
  }

  static constexpr INDEX_TYPE LARGE_NUMBER = 1E6;

  ArrayOfArrays< T > m_array;

  std::vector< std::vector< T > > m_ref;


  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types< INDEX_TYPE, Tensor, TestString >;
TYPED_TEST_CASE( ArrayOfArraysTest, TestTypes );

TYPED_TEST( ArrayOfArraysTest, emptyConstruction )
{
  ArrayOfArrays< TypeParam > a;
  ASSERT_EQ( a.size(), 0 );
  ASSERT_EQ( a.capacity(), 0 );
}


TYPED_TEST( ArrayOfArraysTest, sizedConstruction )
{
  ArrayOfArrays< TypeParam > a( 10 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0 ; i < a.size() ; ++i )
  {
    ASSERT_EQ( a.sizeOfArray( i ), 0 );
    ASSERT_EQ( a.capacityOfArray( i ), 0 );
  }
}

TYPED_TEST( ArrayOfArraysTest, sizedHintConstruction )
{
  ArrayOfArrays< TypeParam > a( 10, 20 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0 ; i < a.size() ; ++i )
  {
    ASSERT_EQ( a.sizeOfArray( i ), 0 );
    ASSERT_EQ( a.capacityOfArray( i ), 20 );
  }
}

TYPED_TEST( ArrayOfArraysTest, appendArray )
{
  this->appendArray( 100, 10 );
}

TYPED_TEST( ArrayOfArraysTest, appendArray2 )
{
  this->appendArray2( 100, 10 );
}

TYPED_TEST( ArrayOfArraysTest, insertArray )
{
  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->insertArray( 100, 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseArray )
{
  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendArray( 100, 10 );
    this->eraseArray( this->m_array.size() / 2 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resize )
{
  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->insertArray( 100, 10 );
    this->resize();
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->resizeArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, appendToArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, atomicAppendToArraySerial )
{
  INDEX_TYPE const APPENDS_PER_ARRAY = 10;

  this->resize( 20, APPENDS_PER_ARRAY );
  this->atomicAppendToArraySerial( APPENDS_PER_ARRAY );
}

#ifdef USE_OPENMP

TYPED_TEST( ArrayOfArraysTest, atomicAppendToArrayOMP )
{
  INDEX_TYPE const APPENDS_PER_THREAD = 10;
  INDEX_TYPE const MAX_TOTAL_APPENDS = APPENDS_PER_THREAD * omp_get_max_threads();

  this->resize( 20, MAX_TOTAL_APPENDS );
  this->atomicAppendToArrayOMP( APPENDS_PER_THREAD );
}

#endif

TYPED_TEST( ArrayOfArraysTest, appendMultipleToArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendMultipleToArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, insertIntoArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->insertIntoArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, insertMultipleIntoArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->insertMultipleIntoArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseFromArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->eraseFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, removeMultipleFromArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->removeMultipleFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, compress )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->compress();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacity )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->resizeArray( 10 );
    this->fill();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacityOfArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->capacityOfArray( 30 );
  }
}

TYPED_TEST( ArrayOfArraysTest, deepCopy )
{
  this->resize( 100 );

  this->appendToArray( 30 );
  this->deepCopy();
}

TYPED_TEST( ArrayOfArraysTest, shallowCopy )
{
  this->resize( 100 );

  this->appendToArray( 30 );
  this->shallowCopy();
}

#ifdef USE_CUDA

template< class T >
using Array1D = Array< T, 1, RAJA::PERM_I, INDEX_TYPE >;

template< class T >
using Array2D = Array< T, 2, RAJA::PERM_IJ, INDEX_TYPE >;

template< class T >
class ArrayOfArraysCudaTest : public ArrayOfArraysTest< T >
{
public:

  void memoryMotion()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();

    // Update the view on the device.
    forall( gpu(), 0, nArrays,
            [view = m_array.toView()] __device__ ( INDEX_TYPE i )
        {
          INDEX_TYPE const sizeOfArray = view.sizeOfArray( i );
          for( INDEX_TYPE j = 0 ; j < sizeOfArray ; ++j )
          {
            LVARRAY_ERROR_IF( &(view[i][j]) != &(view( i, j )), "Value mismatch!" );
            view[i][j] += T( 1 );
            view( i, j ) += T( 1 );
          }
        }
            );

    // Update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      for( INDEX_TYPE j = 0 ; j < m_ref[i].size() ; ++j )
      {
        m_ref[i][j] += T( 1 );
        m_ref[i][j] += T( 1 );
      }
    }

    // Move the view back to the host and compare with the reference.
    forall( sequential(), 0, 1,
            [view = m_array.toView(), this] ( INDEX_TYPE )
    {
      COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
    }
            );
  }

  void memoryMotionMove()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();

    // Update the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toViewC()] __device__ ( INDEX_TYPE i )
        {
          INDEX_TYPE const sizeOfArray = view.sizeOfArray( i );
          for( INDEX_TYPE j = 0 ; j < sizeOfArray ; ++j )
          {
            LVARRAY_ERROR_IF( &(view[i][j]) != &(view( i, j )), "Value mismatch!" );
            view[i][j] += T( 1 );
            view( i, j ) += T( 1 );
          }
        }
            );

    // Update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      for( INDEX_TYPE j = 0 ; j < m_ref[i].size() ; ++j )
      {
        m_ref[i][j] += T( 1 );
        m_ref[i][j] += T( 1 );
      }
    }

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void appendDevice()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToAppend( nArrays );

    // Populate the valuesToAppend array and append the values to the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE maxAppends = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxAppends );

      valuesToAppend[i].resize( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        m_ref[i].push_back( value );
        valuesToAppend[i][j] = value;
      }
    }

    // Append to the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), toAppend=valuesToAppend.toViewConst()] __device__ ( INDEX_TYPE i )
        {
          for( INDEX_TYPE j = 0 ; j < toAppend[i].size() ; ++j )
          {
            view.appendToArray( i, toAppend[i][j] );
          }
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void atomicAppendDevice( INDEX_TYPE const numThreads, INDEX_TYPE const appendsPerArrayPerThread )
  {
    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), 0 );
    }

    forall( gpu(), 0, numThreads,
            [view=m_array.toView(), nArrays, appendsPerArrayPerThread] __device__ ( INDEX_TYPE const threadNum )
        {
          for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
          {
            for( INDEX_TYPE j = 0 ; j < appendsPerArrayPerThread ; ++j )
            {
              T const value = T( i * LARGE_NUMBER + threadNum * appendsPerArrayPerThread + j );
              view.atomicAppendToArray( RAJA::auto_atomic{}, i, value );
            }
          }
        }
            );

    // Now sort each array and check that the values are as expected.
    m_array.move( chai::CPU );
    INDEX_TYPE const appendsPerArray = numThreads * appendsPerArrayPerThread;
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), appendsPerArray );
      ASSERT_LE( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T * const subArrayPtr = m_array[i];
      std::sort( subArrayPtr, subArrayPtr + appendsPerArray );

      for( INDEX_TYPE j = 0 ; j < appendsPerArray ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        EXPECT_EQ( value, m_array( i, j )) << i << ", " << j;
      }
    }
  }

  void appendMultipleDevice()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToAppend( nArrays );

    // Populate the valuesToAppend array and update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxAppends = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxAppends );
      valuesToAppend[i].resize( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        m_ref[i].push_back( value );
        valuesToAppend[i][j] = value;
      }
    }

    // Append to the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), toAppend=valuesToAppend.toViewConst()] __device__ ( INDEX_TYPE i )
        {
          view.appendToArray( i, toAppend[i].data(), toAppend[i].size());
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void insertDevice()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToInsert( nArrays );
    Array1D< Array1D< INDEX_TYPE > > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxInserts );
      valuesToInsert[i].resize( nValues );
      positionsToInsert[i].resize( nValues );

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );

        INDEX_TYPE const insertPos = rand( 0, arraySize + j );
        m_ref[i].insert( m_ref[i].begin() + insertPos, value );
        valuesToInsert[i][j] = value;
        positionsToInsert[i][j] = insertPos;
      }
    }

    // Insert into the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), positions=positionsToInsert.toViewConst(), values=valuesToInsert.toViewConst()]
            __device__ ( INDEX_TYPE i )
        {
          for( INDEX_TYPE j = 0 ; j < values[i].size() ; ++j )
          {
            view.insertIntoArray( i, positions[i][j], values[i][j] );
          }
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void insertMultipleDevice()
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToInsert( nArrays );
    Array1D< INDEX_TYPE > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxInserts );
      valuesToInsert[i].resize( nValues );

      INDEX_TYPE const insertPos = rand( 0, arraySize );
      positionsToInsert[i] = insertPos;

      for( INDEX_TYPE j = 0 ; j < nValues ; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        valuesToInsert[i][j] = value;
      }

      m_ref[i].insert( m_ref[i].begin() + insertPos, valuesToInsert[i].begin(), valuesToInsert[i].end());
    }

    // Insert into the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), positions=positionsToInsert.toViewConst(), values=valuesToInsert.toViewConst()]
            __device__ ( INDEX_TYPE i )
        {
          view.insertIntoArray( i, positions[i], values[i].data(), values[i].size());
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void removeDevice( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< INDEX_TYPE > > positionsToRemove( nArrays );

    // Populate the positionsToRemove array and update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nToRemove = rand( 0, min( arraySize, maxRemoves ));
      positionsToRemove[i].resize( nToRemove );

      for( INDEX_TYPE j = 0 ; j < nToRemove ; ++j )
      {
        INDEX_TYPE const removePosition = rand( 0, arraySize - j - 1 );
        m_ref[i].erase( m_ref[i].begin() + removePosition );
        positionsToRemove[i][j] = removePosition;
      }
    }

    // Remove the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), positions=positionsToRemove.toViewConst()]
            __device__ ( INDEX_TYPE i )
        {
          for( INDEX_TYPE j = 0 ; j < positions[i].size() ; ++j )
          {
            view.eraseFromArray( i, positions[i][j] );
          }
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

  void removeMultipleDevice( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );

    INDEX_TYPE const nArrays = m_array.size();
    Array2D< INDEX_TYPE > removals( nArrays, 2 );

    // Populate the removals array and update the reference.
    for( INDEX_TYPE i = 0 ; i < nArrays ; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      INDEX_TYPE const removePosition = rand( 0, arraySize - 1 );
      INDEX_TYPE const nToRemove = rand( 0, min( maxRemoves, arraySize - removePosition ));

      removals[i][0] = removePosition;
      removals[i][1] = nToRemove;
      m_ref[i].erase( m_ref[i].begin() + removePosition, m_ref[i].begin() + removePosition + nToRemove );
    }

    // Remove from the view on the device.
    forall( gpu(), 0, nArrays,
            [view=m_array.toView(), removals=removals.toViewConst()]
            __device__ ( INDEX_TYPE i )
        {
          if( view.sizeOfArray( i ) == 0 ) return;
          view.eraseFromArray( i, removals[i][0], removals[i][1] );
        }
            );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewCC(), m_ref );
  }

protected:
  using ArrayOfArraysTest< T >::rand;
  using ArrayOfArraysTest< T >::LARGE_NUMBER;
  using ArrayOfArraysTest< T >::m_gen;
  using ArrayOfArraysTest< T >::m_array;
  using ArrayOfArraysTest< T >::m_ref;
};

using CudaTestTypes = ::testing::Types< INDEX_TYPE, Tensor >;
TYPED_TEST_CASE( ArrayOfArraysCudaTest, CudaTestTypes );

TYPED_TEST( ArrayOfArraysCudaTest, memoryMotion )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 30 );
    this->memoryMotion();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, memoryMotionMove )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 30 );
    this->memoryMotionMove();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, append )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->appendDevice();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, atomicAppend )
{
  INDEX_TYPE const NUM_THREADS = 10;
  INDEX_TYPE const APPENDS_PER_THREAD = 10;

  this->resize( 100, NUM_THREADS * APPENDS_PER_THREAD );
  this->atomicAppendDevice( NUM_THREADS, APPENDS_PER_THREAD );
}

TYPED_TEST( ArrayOfArraysCudaTest, appendMultiple )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->appendMultipleDevice();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, insert )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->insertDevice();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, insertMultiple )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->insertMultipleDevice();
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, remove )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->removeDevice( 10 );
  }
}

TYPED_TEST( ArrayOfArraysCudaTest, removeMultiple )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0 ; i < 3 ; ++i )
  {
    this->appendToArray( 10 );
    this->removeMultipleDevice( 10 );
  }
}

#endif // USE_CUDA

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
