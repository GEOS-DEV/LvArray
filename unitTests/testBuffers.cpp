/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file testBuffers.cpp
 */

// Source includes
#include "testUtils.hpp"
#include "bufferManipulation.hpp"
#include "StackBuffer.hpp"
#include "math.hpp"
#include "limits.hpp"

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
                         std::vector< std::remove_const_t< typename BUFFER_TYPE::value_type > > const & ref )
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
using BufferAPITestTypes = ::testing::Types<
  StackBuffer< int, NO_REALLOC_CAPACITY >
  , MallocBuffer< int >
  , MallocBuffer< TestString >
#if defined(LVARRAY_USE_CHAI)
  , ChaiBuffer< int >
  , ChaiBuffer< TestString >
#endif
  >;

TYPED_TEST_SUITE( BufferAPITest, BufferAPITestTypes, );

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

  buffer.moveNested( MemorySpace::host, 0, true );
  buffer.move( MemorySpace::host, true );

  bufferManipulation::free( buffer, 0 );
}

/**
 * @brief Test that the buffer has a registerTouch method.
 */
TYPED_TEST( BufferAPITest, registerTouch )
{
  TypeParam buffer( true );

  buffer.registerTouch( MemorySpace::host );

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

TYPED_TEST( BufferAPITest, getPreviousSpace )
{
  TypeParam buffer( true );

  MemorySpace const previousSpace = buffer.getPreviousSpace();
  LVARRAY_UNUSED_VARIABLE( previousSpace );

  bufferManipulation::free( buffer, 0 );
}

template< typename, typename >
struct ToBufferT
{};

template< typename T, template< typename > class BUFFER, typename U >
struct ToBufferT< T, BUFFER< U > >
{
  using type = BUFFER< T >;
};

template< typename T, typename U, int N >
struct ToBufferT< T, StackBuffer< U, N > >
{
  using type = StackBuffer< T, N >;
};

template< typename BUFFER >
using ToBufferConst = typename ToBufferT< typename BUFFER::value_type const, BUFFER >::type;

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
    m_buffer.reallocate( 0, MemorySpace::host, NO_REALLOC_CAPACITY );
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
   * @brief Test emplaceBack.
   * @param nVals the number of values to append.
   */
  void emplaceBack( std::ptrdiff_t const nVals )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    LVARRAY_ERROR_IF_NE( nVals % 2, 0 );

    for( std::ptrdiff_t i = 0; i < nVals / 2; ++i )
    {
      T const val( randInt() );
      bufferManipulation::emplaceBack( m_buffer, size(), val );
      m_ref.emplace_back( val );

      std::ptrdiff_t const seed = randInt();
      bufferManipulation::emplaceBack( m_buffer, size(), seed );
      m_ref.emplace_back( seed );
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test emplace.
   * @param nToInsert the number of values to insert.
   */
  void emplace( std::ptrdiff_t const nToInsert )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    LVARRAY_ERROR_IF_NE( nToInsert % 2, 0 );

    for( std::ptrdiff_t i = 0; i < nToInsert / 2; ++i )
    {
      {
        T const val( randInt() );
        std::ptrdiff_t const position = randInt( size() );
        bufferManipulation::emplace( m_buffer, size(), position, val );
        m_ref.emplace( m_ref.begin() + position, val );
      }

      {
        std::ptrdiff_t const seed = randInt();
        std::ptrdiff_t const position = randInt( size() );
        bufferManipulation::emplace( m_buffer, size(), position, seed );
        m_ref.emplace( m_ref.begin() + position, seed );
      }
    }

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test insert which inserts multiple values.
   * @param nToInsert the number of values to insert.
   */
  void insert( std::ptrdiff_t const nToInsert )
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    constexpr std::ptrdiff_t MAX_VALS_PER_INSERT = 20;

    std::ptrdiff_t nInserted = 0;
    while( nInserted < nToInsert )
    {
      // Insert from a std::vector
      {
        std::ptrdiff_t const nToInsertThisIter = randInt( math::min( MAX_VALS_PER_INSERT, nToInsert - nInserted ) );
        std::vector< T > valsToInsert( nToInsertThisIter );

        for( T & val : valsToInsert )
        { val = T( randInt() ); }

        std::ptrdiff_t const position = randInt( size() );
        bufferManipulation::insert( m_buffer, size(), position, valsToInsert.begin(), valsToInsert.end() );
        m_ref.insert( m_ref.begin() + position, valsToInsert.begin(), valsToInsert.end() );

        nInserted += nToInsertThisIter;
      }

      // Insert from a std::list
      {
        std::ptrdiff_t const nToInsertThisIter = randInt( math::min( MAX_VALS_PER_INSERT, nToInsert - nInserted ) );
        std::list< T > valsToInsert( nToInsertThisIter );
        for( T & val : valsToInsert )
        { val = T( randInt() ); }

        std::ptrdiff_t const position = randInt( size() );
        bufferManipulation::insert( m_buffer, size(), position, valsToInsert.begin(), valsToInsert.end() );
        m_ref.insert( m_ref.begin() + position, valsToInsert.begin(), valsToInsert.end() );

        nInserted += nToInsertThisIter;
      }
    }

    LVARRAY_ERROR_IF_NE( nInserted, nToInsert );
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


TYPED_TEST_SUITE( BufferTestNoRealloc, BufferAPITestTypes, );

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
 * @brief Test bufferManipulation::emplaceBack without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, emplaceBack )
{
  this->emplaceBack( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test the copy constructor.
 */
TYPED_TEST( BufferTestNoRealloc, CopyConstructor )
{
  this->emplaceBack( 100 );
  TypeParam copy( this->m_buffer, 100 );
  TypeParam copy2( copy );

  EXPECT_EQ( this->m_buffer.capacity(), copy.capacity() );
  EXPECT_EQ( this->m_buffer.capacity(), copy2.capacity() );


  if( TypeParam::hasShallowCopy )
  {
    EXPECT_EQ( this->m_buffer.data(), copy.data() );
    EXPECT_EQ( this->m_buffer.data(), copy2.data() );
  }
  else
  {
    EXPECT_NE( this->m_buffer.data(), copy.data() );
    EXPECT_NE( this->m_buffer.data(), copy2.data() );
  }

  COMPARE_TO_REFERENCE( copy, this->m_ref );
  COMPARE_TO_REFERENCE( copy2, this->m_ref );
}

/**
 * @brief Test the copy assignment operator.
 */
TYPED_TEST( BufferTestNoRealloc, copyAssignmentOperator )
{
  this->emplaceBack( 100 );
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
  this->emplaceBack( 100 );
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
  this->emplaceBack( 100 );
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
 * @brief Test the construction of buffer of T const.
 */
TYPED_TEST( BufferTestNoRealloc, ConstConstructor )
{
  this->emplaceBack( 100 );
  ToBufferConst< TypeParam > copy( this->m_buffer );

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
  COMPARE_TO_REFERENCE( this->m_buffer, this->m_ref );
}

/**
 * @brief Test bufferManipulation::popBack without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, popBack )
{
  this->emplaceBack( NO_REALLOC_CAPACITY );
  this->popBack( NO_REALLOC_CAPACITY / 2 );
}

/**
 * @brief Test bufferManipulation::emplace without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, emplace )
{
  this->emplace( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::insert multiple function without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, insert )
{
  this->insert( NO_REALLOC_CAPACITY );
}

/**
 * @brief Test bufferManipulation::erase without reallocating.
 */
TYPED_TEST( BufferTestNoRealloc, erase )
{
  this->emplaceBack( NO_REALLOC_CAPACITY );
  this->erase( NO_REALLOC_CAPACITY / 2 );
}

/**
 * @brief Test bufferManipulation::copyInto with each of the buffer types.
 */
TYPED_TEST( BufferTestNoRealloc, copyInto )
{
  this->emplaceBack( NO_REALLOC_CAPACITY );

  typeManipulation::forEachArg(
    [this]( auto && copy )
  {
    bufferManipulation::copyInto( copy, 0, this->m_buffer, NO_REALLOC_CAPACITY );
    COMPARE_TO_REFERENCE( copy, this->m_ref );
    bufferManipulation::free( copy, NO_REALLOC_CAPACITY );
  },
  #if defined(LVARRAY_USE_CHAI)
    ChaiBuffer< typename TypeParam::value_type >( true ),
  #endif
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

    bufferManipulation::setCapacity( m_buffer, size(), MemorySpace::host, size() + 100 );
    T const * const ptr = m_buffer.data();

    T const val( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 100, val );
    m_ref.resize( size() + 100, val );

    EXPECT_EQ( ptr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::setCapacity( m_buffer, size(), MemorySpace::host, size() - 50 );
    m_ref.resize( integerConversion< std::size_t >( size() - 50 ) );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  /**
   * @brief Test reserve.
   */
  void reserve()
  {
    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::setCapacity( m_buffer, size(), MemorySpace::host, size() + 100 );
    T const * const ptr = m_buffer.data();

    T const val( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 100, val );
    m_ref.resize( size() + 100, val );

    bufferManipulation::reserve( m_buffer, size(), MemorySpace::host, size() - 50 );

    EXPECT_EQ( ptr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );

    bufferManipulation::reserve( m_buffer, size(), MemorySpace::host, size() + 50 );
    T const * const newPtr = m_buffer.data();

    T const newVal( randInt() );
    bufferManipulation::resize( m_buffer, size(), size() + 50, newVal );
    m_ref.resize( size() + 50, newVal );

    EXPECT_EQ( newPtr, m_buffer.data() );

    COMPARE_TO_REFERENCE( m_buffer, m_ref );
  }

  void typeConversion()
  {
    this->resize( 60 );

    typename ToBufferT< T[ 4 ], BUFFER_TYPE >::type buffer4( m_buffer );
    EXPECT_EQ( buffer4.capacity(), m_buffer.capacity() / 4 );

    if( BUFFER_TYPE::hasShallowCopy )
    {
      EXPECT_EQ( static_cast< void * >( buffer4.data() ), static_cast< void * >( m_buffer.data() ) );
    }

    for( std::ptrdiff_t i = 0; i < size() / 4; ++i )
    {
      for( std::ptrdiff_t j = 0; j < 4; ++j )
      {
        EXPECT_EQ( buffer4[ i ][ j ], m_buffer[ 4 * i + j ] );
      }
    }

    typename ToBufferT< T[ 2 ], BUFFER_TYPE >::type buffer2( buffer4 );
    EXPECT_EQ( buffer2.capacity(), m_buffer.capacity() / 2 );

    if( BUFFER_TYPE::hasShallowCopy )
    {
      EXPECT_EQ( static_cast< void * >( buffer2.data() ), static_cast< void * >( m_buffer.data() ) );
    }

    for( std::ptrdiff_t i = 0; i < size() / 2; ++i )
    {
      for( std::ptrdiff_t j = 0; j < 2; ++j )
      {
        EXPECT_EQ( buffer2[ i ][ j ], m_buffer[ 2 * i + j ] );
      }
    }
  }
};

/// The list of types to instantiate BufferTestWithRealloc with,
/// should contain at least one instance of every dynamically reallocatable buffer class.
using BufferTestWithReallocTypes = ::testing::Types<
  MallocBuffer< int >
  , MallocBuffer< TestString >
#if defined(LVARRAY_USE_CHAI)
  , ChaiBuffer< int >
  , ChaiBuffer< TestString >
#endif
  >;

TYPED_TEST_SUITE( BufferTestWithRealloc, BufferTestWithReallocTypes, );

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
 * @brief Test bufferManipulation::emplaceBack.
 */
TYPED_TEST( BufferTestWithRealloc, emplaceBack )
{
  this->emplaceBack( 100 );
  this->emplaceBack( 100 );
  this->emplaceBack( 100 );
}

/**
 * @brief Test bufferManipulation::emplace.
 */
TYPED_TEST( BufferTestWithRealloc, emplace )
{
  this->emplace( 100 );
  this->emplace( 100 );
  this->emplace( 100 );
}

/**
 * @brief Test bufferManipulation::insert that inserts multiple values.
 */
TYPED_TEST( BufferTestWithRealloc, insert )
{
  this->insert( 100 );
  this->insert( 100 );
  this->insert( 100 );
}

/**
 * @brief Test multiple bufferManipulation methods chained together.
 */
TYPED_TEST( BufferTestWithRealloc, combination )
{
  this->emplaceBack( 100 );
  this->emplace( 100 );
  this->erase( 100 );
  this->insert( 100 );
  this->popBack( 100 );
  this->emplaceBack( 100 );
  this->emplace( 100 );
  this->erase( 100 );
  this->insert( 100 );
  this->popBack( 100 );
}

/**
 * @brief Test buffer type conversion.
 */
TYPED_TEST( BufferTestWithRealloc, typeConversion )
{
  this->typeConversion();
}

// TODO:
// BufferTestNoRealloc on device with StackBuffer + MallocBuffer

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
