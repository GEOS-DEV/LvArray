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

/**
 * @file testBuffers.cpp
 */

// Source includes
#include "testUtils.hpp"
#include "bufferManipulation.hpp"
#include "ChaiBuffer.hpp"
#include "NewChaiBuffer.hpp"
#include "StackBuffer.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <vector>
#include <random>


namespace LvArray
{
namespace testing
{

/**
 * @tparam BUFFER_TYPE the type of the buffer to check.
 * @brief Check that the provided buffer has identical contents to the std::vector.
 * @param buffer the buffer to check, the size is given by ref.
 * @param ref the reference std::vector to compare against.
 */
template< typename BUFFER_TYPE >
void compareToReference( BUFFER_TYPE const & buffer,
                         std::vector< typename BUFFER_TYPE::value_type > const & ref )
{
  std::ptrdiff_t const size = ref.size();
  ASSERT_GE( buffer.capacity(), size );

  typename BUFFER_TYPE::value_type const * const ptr = buffer.data();
  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    ASSERT_EQ( &buffer[ i ], ptr + i );
    EXPECT_EQ( buffer[ i ], ref[ i ] );
  }
}

// Use this macro to call compareToReference with better error messages.
#define COMPARE_TO_REFERENCE( buffer, ref ) { SCOPED_TRACE( "" ); compareToReference( buffer, ref ); \
}

constexpr std::ptrdiff_t NO_REALLOC_CAPACITY = 100;

/**
 * @brief Placeholder class for testing the expected API of the various Buffer classes.
 */
template< typename BUFFER_TYPE >
class BufferAPITest : public ::testing::Test
{};

/// The list of types to instantiate BufferAPITest with, should contain at least one instance of every buffer class.
using BufferAPITestTypes = ::testing::Types< ChaiBuffer< int >,
                                             ChaiBuffer< TestString >,
                                             NewChaiBuffer< int >,
                                             NewChaiBuffer< TestString >,
                                             StackBuffer< int, NO_REALLOC_CAPACITY >,
                                             MallocBuffer< int >,
                                             MallocBuffer< TestString >
                                             >;

TYPED_TEST_CASE( BufferAPITest, BufferAPITestTypes );

/**
 * @brief Test that the buffer has a static constexpr hasShallowCopy member.
 */
TYPED_TEST( BufferAPITest, hasShallowCopy )
{
  constexpr bool hasShallowCopy = TypeParam::hasShallowCopy;
  LVARRAY_UNUSED_VARIABLE( hasShallowCopy );
}

/**
 * @brief Function to silence unused warnings with nvcc where casting to void doesn't work.
 */
template< typename T >
void voidFunction( T & )
{}

/**
 * @brief Test that the buffer is default constructable and that no memory
 *        leak occurs when you don't free it.
 */
TYPED_TEST( BufferAPITest, DefaultConstruction )
{
  TypeParam buffer;
  voidFunction( buffer );
}

/**
 * @brief Test the owning constructor.
 */
TYPED_TEST( BufferAPITest, OwningConstruction )
{
  TypeParam buffer( true );
  bufferManipulation::free( buffer, 0 );
}

/**
 * @brief Test that the buffer has a move method.
 */
TYPED_TEST( BufferAPITest, move )
{
  TypeParam buffer( true );

  buffer.move( chai::CPU );
  buffer.move( chai::CPU, true );

  bufferManipulation::free( buffer, 0 );
}

/**
 * @brief Test that the buffer has a registerTouch method.
 */
TYPED_TEST( BufferAPITest, registerTouch )
{
  TypeParam buffer( true );

  buffer.registerTouch( chai::CPU );

  bufferManipulation::free( buffer, 0 );
}

/**
 * @brief Test that the buffer has a setName method.
 */
TYPED_TEST( BufferAPITest, setName )
{
  TypeParam buffer( true );

  buffer.setName( "dummy" );

  bufferManipulation::free( buffer, 0 );
}

/**
 * @tparam BUFFER_TYPE the type of the Buffer to test.
 * @brief Testing class that holds a buffer, size and a std::vector to compare against.
 */
template< typename BUFFER_TYPE >
class BufferTestNoRealloc : public ::testing::Test
{
public:

  /// Alias for the type stored in BUFFER_TYPE.
  using T = typename BUFFER_TYPE::value_type;

  /// Default constructor.
  BufferTestNoRealloc():
    m_buffer( true ),
    m_ref()
  {};

  /**
   * @brief googletest SetUp method which gets called right after the constructor, allocates space in m_buffer.
   *        This logic is placed here because it is overwritten in derived classes.
   */
  void SetUp() override
  {
    m_buffer.reallocate( 0, NO_REALLOC_CAPACITY );
  }

  /**
   * @brief Destructor, free's the data in m_buffer.
   */
  ~BufferTestNoRealloc()
  {
    bufferManipulation::free( m_buffer, size() );
  }

  std::ptrdiff_t size() const
  {
    return m_ref.size();
  }

  /**
   * @brief Test resize.
   * @param newSize the size to resize m_buffer to.
   */
  void resize( std::ptrdiff_t const newSize )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    T const val( randInt() );
    bufferManipulation::resize( m_buffer, size(), newSize, val );
    m_ref.resize( newSize, val );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    for( std::ptrdiff_t i = 0; i < size(); ++i )
    {
      T const v( randInt() );
      m_buffer[ i ] = v;
      m_ref[ i ] = v;
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test pushBack.
   * @param nVals the number of values to append.
   */
  void pushBack( std::ptrdiff_t const nVals )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    for( std::ptrdiff_t i = 0; i < nVals; ++i )
    {
      T const val( randInt() );
      bufferManipulation::pushBack( m_buffer, size(), val );
      m_ref.push_back( val );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test pushBack which takes an r-value reference.
   * @param nVals the number of values to append.
   */
  void pushBackMove( std::ptrdiff_t const nVals )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    for( std::ptrdiff_t i = 0; i < nVals; ++i )
    {
      std::ptrdiff_t const seed = randInt();
      bufferManipulation::pushBack( m_buffer, size(), T( seed ) );
      m_ref.push_back( T( seed ) );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test popBack.
   * @param nToPop the number of values to pop.
   */
  void popBack( std::ptrdiff_t const nToPop )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    LVARRAY_ERROR_IF_GT( nToPop, size() );

    for( std::ptrdiff_t i = 0; i < nToPop; ++i )
    {
      bufferManipulation::popBack( m_buffer, size() );
      m_ref.pop_back();
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test insert.
   * @param nToInsert the number of values to insert.
   */
  void insert( std::ptrdiff_t const nToInsert )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    for( std::ptrdiff_t i = 0; i < nToInsert; ++i )
    {
      T const val( randInt() );
      std::ptrdiff_t const position = randInt( size() );
      bufferManipulation::insert( m_buffer, size(), position, val );
      m_ref.insert( m_ref.begin() + position, val );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test insert which takes an r-value reference.
   * @param nToInsert the number of values to insert.
   */
  void insertMove( std::ptrdiff_t const nToInsert )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    for( std::ptrdiff_t i = 0; i < nToInsert; ++i )
    {
      std::ptrdiff_t const seed = randInt();
      std::ptrdiff_t const position = randInt( size() );
      bufferManipulation::insert( m_buffer, size(), position, T( seed ) );
      m_ref.insert( m_ref.begin() + position, T( seed ) );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test insert which inserts multiple values.
   * @param nToInsert the number of values to insert.
   */
  void insertMultiple( std::ptrdiff_t const nToInsert )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    constexpr std::ptrdiff_t MAX_VALS_PER_INSERT = 20;
    std::vector< T > valsToInsert( MAX_VALS_PER_INSERT );

    std::ptrdiff_t nInserted = 0;
    while( nInserted < nToInsert )
    {
      std::ptrdiff_t const nToInsertThisIter = randInt( min( MAX_VALS_PER_INSERT, nToInsert - nInserted ) );
      std::ptrdiff_t const position = randInt( size() );

      for( std::ptrdiff_t i = 0; i < nToInsertThisIter; ++i )
      {
        valsToInsert[ i ] = T( randInt() );
      }

      bufferManipulation::insert( m_buffer, size(), position, valsToInsert.data(), nToInsertThisIter );
      m_ref.insert( m_ref.begin() + position, valsToInsert.begin(), valsToInsert.begin() + nToInsertThisIter );

      nInserted += nToInsertThisIter;
    }

    LVARRAY_ERROR_IF_NE( nInserted, nToInsert );
    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test erase.
   * @param nToInsert the number of values to erase.
   */
  void erase( std::ptrdiff_t const nToErase )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    LVARRAY_ERROR_IF_GT( nToErase, size() );

    for( std::ptrdiff_t i = 0; i < nToErase; ++i )
    {
      std::ptrdiff_t const position = randInt( size() - 1 );
      bufferManipulation::erase( m_buffer, size(), position );
      m_ref.erase( m_ref.begin() + position );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

protected:

  /**
   * @brief Return a random integer in [0, max].
   */
  std::ptrdiff_t randInt( std::ptrdiff_t const max = 1000 )
  {
    return std::uniform_int_distribution< std::ptrdiff_t >( 0, max )( m_gen );
  }

  /// Random number generator for each test case.
  std::mt19937_64 m_gen;

  /// The buffer to test.
  BUFFER_TYPE m_buffer;

  /// The reference to compare against.
  std::vector< T > m_ref;
};


TYPED_TEST_CASE( BufferTestNoRealloc, BufferAPITestTypes );

/**
 * @brief Test bufferManipulation::resize without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, resize )
{
  this->resize( NO_REALLOC_CAPACITY / 2 );
  this->resize( NO_REALLOC_CAPACITY / 4 );
  this->resize( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::pushBack without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, pushBack )
{
  this->pushBack( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test the copy constructor.
 */
TYPED_TEST( BufferTestNoRealloc, CopyConstructor )
{
  this->pushBack( 100 );
  TypeParam copy( this->m_buffer );
  EXPECT_EQ( this->m_buffer.capacity(), copy.capacity() );

  if( TypeParam::hasShallowCopy )
  {
    EXPECT_EQ( this->m_buffer.data(), copy.data() );
  }
  else
  {
    EXPECT_NE( this->m_buffer.data(), copy.data() );
  }

  COMPARE_TO_REFERENCE( copy, this->m_ref );
}

/**
 * @brief Test the copy assignment operator.
 */
TYPED_TEST( BufferTestNoRealloc, copyAssignmentOperator )
{
  this->pushBack( 100 );
  TypeParam copy;
  copy = this->m_buffer;
  EXPECT_EQ( this->m_buffer.capacity(), copy.capacity() );

  if( TypeParam::hasShallowCopy )
  {
    EXPECT_EQ( this->m_buffer.data(), copy.data() );
  }
  else
  {
    EXPECT_NE( this->m_buffer.data(), copy.data() );
  }

  COMPARE_TO_REFERENCE( copy, this->m_ref );
}

/**
 * @brief Test the move constructor.
 */
TYPED_TEST( BufferTestNoRealloc, MoveConstructor )
{
  this->pushBack( 100 );
  TypeParam copy( std::move( this->m_buffer ) );
  COMPARE_TO_REFERENCE( copy, this->m_ref );

  bufferManipulation::free( copy, this->size() );

  // This is kind of a hack. We free the copy above because m_buffer has
  // been invalidated by the std::move. However the BufferTestNoRealloc
  // destructor will attempt to destroy the values in m_buffer between
  // 0 and m_ref.size(), so we set the size of m_ref to 0 to prevent this.
  this->m_ref.clear();
}

/**
 * @brief Test the move assignment operator.
 */
TYPED_TEST( BufferTestNoRealloc, moveAssignmentOperator )
{
  this->pushBack( 100 );
  TypeParam copy;
  copy = std::move( this->m_buffer );
  COMPARE_TO_REFERENCE( copy, this->m_ref );

  bufferManipulation::free( copy, this->size() );

  // This is kind of a hack. We free the copy above because m_buffer has
  // been invalidated by the std::move. However the BufferTestNoRealloc
  // destructor will attempt to destroy the values in m_buffer between
  // 0 and m_ref.size(), so we set the size of m_ref to 0 to prevent this.
  this->m_ref.clear();
}

/**
 * @brief Test bufferManipulation::pushBack that takes an rvalue-reference
 *        without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, pushBackMove )
{
  this->pushBackMove( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::popBack without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, popBack )
{
  this->pushBack( NO_REALLOC_CAPACITY );
  this->popBack( NO_REALLOC_CAPACITY / 2 );
}

/**
 * @brief Test bufferManipulation::insert without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, insert )
{
  this->insert( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::insert that takes an rvalue-reference
 *        without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, insertMove )
{
  this->insertMove( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::insert multiple function without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, insertMultiple )
{
  this->insertMultiple( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::erase without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, erase )
{
  this->pushBack( NO_REALLOC_CAPACITY );
  this->erase( NO_REALLOC_CAPACITY / 2 );
}

/**
 * @brief Test bufferManipulation::copyInto with each of the buffer types.
 */
TYPED_TEST( BufferTestNoRealloc, copyInto )
{
  this->pushBack( NO_REALLOC_CAPACITY );

  for_each_arg(
    [this]( auto && copy )
  {
    bufferManipulation::copyInto( copy, 0, this->m_buffer, NO_REALLOC_CAPACITY );
    COMPARE_TO_REFERENCE( copy, this->m_ref );
    bufferManipulation::free( copy, NO_REALLOC_CAPACITY );
  },
    ChaiBuffer< typename TypeParam::value_type >( true ),
    NewChaiBuffer< typename TypeParam::value_type >( true ),
    MallocBuffer< typename TypeParam::value_type >( true )
    // I would like to copInto a StackBuffer but that doesn't work with std::string.
    );

  COMPARE_TO_REFERENCE( this->m_buffer, this->m_ref );
}

/**
 * @tparam BUFFER_TYPE the type of the Buffer to test.
 * @brief Testing class for testing buffers with reallocation involved.
 */
template< typename BUFFER_TYPE >
class BufferTestWithRealloc : public BufferTestNoRealloc< BUFFER_TYPE >
{
public:

  /// Alias for the type stored in BUFFER_TYPE.
  using T = typename BUFFER_TYPE::value_type;

  /// Aliases for the parent class members and methods.
  using BufferTestNoRealloc< BUFFER_TYPE >::m_buffer;
  using BufferTestNoRealloc< BUFFER_TYPE >::m_ref;
  using BufferTestNoRealloc< BUFFER_TYPE >::size;
  using BufferTestNoRealloc< BUFFER_TYPE >::randInt;

  /**
   * @brief Override the parent class SetUp.
   */
  void SetUp() override
  {}

  /**
   * @brief Test setCapacity.
   */
  void setCapacity()
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::setCapacity( m_buffer, size(), size() + 100 );
    T const * const ptr = m_buffer.data();

    T const val( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 100, val );
    m_ref.resize( size() + 100, val );

    EXPECT_EQ( ptr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::setCapacity( m_buffer, size(), size() - 50 );
    m_ref.resize( size() - 50 );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test setCapacity.
   */
  void reserve()
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::setCapacity( m_buffer, size(), size() + 100 );
    T const * const ptr = m_buffer.data();

    T const val( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 100, val );
    m_ref.resize( size() + 100, val );

    bufferManipulation::reserve( m_buffer, size(), size() - 50 );

    EXPECT_EQ( ptr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::reserve( m_buffer, size(), size() + 50 );
    T const * const newPtr = m_buffer.data();

    T const newVal( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 50, newVal );
    m_ref.resize( size() + 50, newVal );

    EXPECT_EQ( newPtr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }
};

/// The list of types to instantiate BufferTestWithRealloc with,
/// should contain at least one instance of every dynamically reallocatable buffer class.
using BufferTestWithReallocTypes = ::testing::Types< ChaiBuffer< int >,
                                                     ChaiBuffer< TestString >,
                                                     MallocBuffer< int >,
                                                     MallocBuffer< TestString >
                                                     >;

TYPED_TEST_CASE( BufferTestWithRealloc, BufferTestWithReallocTypes );

/**
 * @brief Test bufferManipulation::setCapacity.
 */
TYPED_TEST( BufferTestWithRealloc, setCapacity )
{
  this->setCapacity();
}

/**
 * @brief Test bufferManipulation::reserve.
 */
TYPED_TEST( BufferTestWithRealloc, reserve )
{
  this->reserve();
}

/**
 * @brief Test bufferManipulation::pushBack.
 */
TYPED_TEST( BufferTestWithRealloc, pushBack )
{
  this->pushBack( 100 );
  this->pushBack( 100 );
  this->pushBack( 100 );
}

/**
 * @brief Test bufferManipulation::pushBack that takes an r-value reference.
 */
TYPED_TEST( BufferTestWithRealloc, pushBackMove )
{
  this->pushBackMove( 100 );
  this->pushBackMove( 100 );
  this->pushBackMove( 100 );
}

/**
 * @brief Test bufferManipulation::insert.
 */
TYPED_TEST( BufferTestWithRealloc, insert )
{
  this->insert( 100 );
  this->insert( 100 );
  this->insert( 100 );
}

/**
 * @brief Test bufferManipulation::insert that takes an r-value reference.
 */
TYPED_TEST( BufferTestWithRealloc, insertMove )
{
  this->insertMove( 100 );
  this->insertMove( 100 );
  this->insertMove( 100 );
}

/**
 * @brief Test bufferManipulation::insert that inserts multiple values.
 */
TYPED_TEST( BufferTestWithRealloc, insertMultiple )
{
  this->insertMultiple( 100 );
  this->insertMultiple( 100 );
  this->insertMultiple( 100 );
}

/**
 * @brief Test multiple bufferManipulation methods chained together.
 */
TYPED_TEST( BufferTestWithRealloc, combination )
{
  this->pushBack( 100 );
  this->insert( 100 );
  this->erase( 100 );
  this->insertMultiple( 100 );
  this->popBack( 100 );
  this->pushBack( 100 );
  this->insert( 100 );
  this->erase( 100 );
  this->insertMultiple( 100 );
  this->popBack( 100 );
}


// TODO:
// BufferTestNoRealloc on device with StackBuffer + MallocBuffer
// Move tests with ChaiBuffer
// Copy-capture tests with ChaiBuffer

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
