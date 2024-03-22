/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "output.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

inline INDEX_TYPE randomInteger( INDEX_TYPE const min, INDEX_TYPE const max )
{
  static std::mt19937_64 gen;
  return std::uniform_int_distribution< INDEX_TYPE >( min, max )( gen );
}

template< typename ARRAY >
class ArrayTest : public ::testing::Test
{
public:
  using T = typename ARRAY::ValueType;
  static constexpr int NDIM = ARRAY::NDIM;
  using PERMUTATION = typename ARRAY::Permutation;
  static constexpr int USD = ARRAY::USD;
  using INDEX_TYPE = typename ARRAY::IndexType;

  static void defaultConstructor()
  {
    ARRAY array;
    EXPECT_TRUE( array.empty() );

    for( int i = 0; i < NDIM; ++i )
    {
      EXPECT_EQ( array.size( i ), 0 );
    }

    EXPECT_EQ( array.size(), 0 );
    EXPECT_EQ( array.capacity(), 0 );
    EXPECT_EQ( array.data(), nullptr );
  }

  static std::unique_ptr< ARRAY > sizedConstructor()
  {
    std::array< INDEX_TYPE, NDIM > sizes;
    for( int dim = 0; dim < NDIM; ++dim )
    { sizes[ dim ] = randomInteger( 1, getMaxDimSize() ); }

    std::unique_ptr< ARRAY > array = forwardArrayAsArgs( sizes,
                                                         [] ( auto const ... args )
    { return std::make_unique< ARRAY >( args ... ); } );
    EXPECT_FALSE( array->empty() );

    INDEX_TYPE totalSize = 1;
    for( int i = 0; i < NDIM; ++i )
    {
      EXPECT_EQ( array->size( i ), sizes[ i ] );
      totalSize *= sizes[ i ];
    }

    EXPECT_EQ( array->size(), totalSize );
    EXPECT_EQ( array->capacity(), totalSize );
    EXPECT_NE( array->data(), nullptr );

    fill( *array );

    return array;
  }

  static void copyConstructor()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    ARRAY copy = *array;
    compare( *array, copy );
  }

  static void moveConstructor()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    ARRAY copy = *array;
    ARRAY movedCopy = std::move( *array );

    EXPECT_TRUE( array->empty() );
    EXPECT_EQ( array->size(), 0 );
    EXPECT_EQ( array->capacity(), 0 );
    EXPECT_EQ( array->data(), nullptr );

    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( array->size( dim ), 0 );
    }

    array.reset();

    compare( copy, movedCopy );
  }

  static void toView()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    compare( *array, array->toView(), true );
  }

  static void toViewConst()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    compare( *array, array->toViewConst(), true );
  }

  static void copyAssignmentOperator()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    ARRAY copy;
    EXPECT_TRUE( copy.empty() );

    copy = *array;
    compare( *array, copy );
  }

  static void moveAssignmentOperator()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    ARRAY copy = *array;
    ARRAY movedCopy;
    EXPECT_TRUE( movedCopy.empty() );

    movedCopy = std::move( *array );

    EXPECT_TRUE( array->empty() );
    EXPECT_EQ( array->size(), 0 );
    EXPECT_EQ( array->capacity(), 0 );
    EXPECT_EQ( array->data(), nullptr );

    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( array->size( dim ), 0 );
    }

    array.reset();

    compare( copy, movedCopy );
  }

  static void resizeFromPointer()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();

    std::array< INDEX_TYPE, NDIM > newSizes;
    std::array< INDEX_TYPE, NDIM > oldSizes;

    for( int dim = 0; dim < NDIM; ++dim )
    {
      oldSizes[ dim ] = array->size( dim );
      newSizes[ dim ] = randomInteger( 1, getMaxDimSize() );
    }
    array->resize( NDIM, newSizes.data() );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, true );

    for( int dim = 0; dim < NDIM; ++dim )
    {
      oldSizes[ dim ] = array->size( dim );
      newSizes[ dim ] = randomInteger( 1, getMaxDimSize() );
    }
    array->resize( NDIM, newSizes.data() );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, true );
  }

  static void resizeFromArgs()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();

    std::array< INDEX_TYPE, NDIM > newSizes;
    std::array< INDEX_TYPE, NDIM > oldSizes;

    for( int dim = 0; dim < NDIM; ++dim )
    {
      oldSizes[ dim ] = array->size( dim );
      newSizes[ dim ] = randomInteger( 1, getMaxDimSize() );
    }
    forwardArrayAsArgs( newSizes, [&array]( auto const ... indices )
    {
      return array->resize( indices ... );
    } );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, true );

    // Increase the size of every dimension.
    for( int dim = 0; dim < NDIM; ++dim )
    {
      oldSizes[ dim ] = array->size( dim );
      newSizes[ dim ] = randomInteger( 1, getMaxDimSize() );
    }
    forwardArrayAsArgs( newSizes, [&array]( auto const ... indices )
    {
      return array->resize( indices ... );
    } );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, true );
  }

  template< int _NDIM=NDIM >
  static std::enable_if_t< _NDIM == 1 >
  resizeDimension()
  {
    ARRAY array( 10 );
    fill( array );

    resizeOneDimension< 0 >( array );
  }

  template< int _NDIM=NDIM >
  static std::enable_if_t< _NDIM == 2 >
  resizeDimension()
  {
    ARRAY array( 20, 20 );
    fill( array );

    resizeOneDimension< 0 >( array );
    resizeOneDimension< 1 >( array );

    resizeTwoDimensions< 0, 1 >( array );
  }

  template< int _NDIM=NDIM >
  static std::enable_if_t< _NDIM == 3 >
  resizeDimension()
  {
    ARRAY array( 8, 8, 8 );
    fill( array );

    resizeOneDimension< 0 >( array );
    resizeOneDimension< 1 >( array );
    resizeOneDimension< 2 >( array );

    resizeTwoDimensions< 0, 1 >( array );
    resizeTwoDimensions< 0, 2 >( array );
    resizeTwoDimensions< 1, 2 >( array );

    resizeThreeDimensions< 0, 1, 2 >( array );
  }

  template< int _NDIM=NDIM >
  static std::enable_if_t< _NDIM == 4 >
  resizeDimension()
  {
    ARRAY array( 3, 3, 3, 3 );
    fill( array );

    resizeOneDimension< 0 >( array );
    resizeOneDimension< 1 >( array );
    resizeOneDimension< 2 >( array );
    resizeOneDimension< 3 >( array );

    resizeTwoDimensions< 0, 1 >( array );
    resizeTwoDimensions< 0, 2 >( array );
    resizeTwoDimensions< 0, 3 >( array );
    resizeTwoDimensions< 1, 2 >( array );
    resizeTwoDimensions< 1, 3 >( array );
    resizeTwoDimensions< 2, 3 >( array );

    resizeThreeDimensions< 0, 1, 2 >( array );
    resizeThreeDimensions< 0, 1, 3 >( array );
    resizeThreeDimensions< 0, 2, 3 >( array );
    resizeThreeDimensions< 1, 2, 3 >( array );
  }

  static void resize( bool const useDefault )
  {
    ARRAY array;

    INDEX_TYPE const maxDimSize = getMaxDimSize();

    // Iterate over randomly generated sizes.
    for( int i = 0; i < 10; ++i )
    {
      // Default dimension is the first dimension
      int constexpr dim = 0;

      std::array< INDEX_TYPE, NDIM > oldSizes;
      for( int d = 0; d < NDIM; ++d )
      { oldSizes[ d ] = randomInteger( 1, maxDimSize / 2 ); }

      std::array< INDEX_TYPE, NDIM > newSizes = oldSizes;

      array.resize( NDIM, oldSizes.data() );

      fill( array );

      // Increase the size
      newSizes[ dim ] = randomInteger( oldSizes[ dim ], maxDimSize );
      if( useDefault )
      {
        array.resizeDefault( newSizes[ dim ], T( -i * dim ) );
        checkResize( array, oldSizes, newSizes, true, true, T( -i * dim ) );
      }
      else
      {
        array.resize( newSizes[ dim ] );
        checkResize( array, oldSizes, newSizes, true, true );
      }
      oldSizes = newSizes;

      // Decrease the size
      newSizes[ dim ] = randomInteger( 0, oldSizes[ dim ] );
      if( useDefault )
      {
        array.resizeDefault( newSizes[ dim ], T( -i * dim - 1 ) );
        checkResize( array, oldSizes, newSizes, true, true, T( -i * dim -1 ) );
      }
      else
      {
        array.resize( newSizes[ dim ] );
        checkResize( array, oldSizes, newSizes, true, true );
      }
    }
  }

  static void reserveAndCapacity()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();
    INDEX_TYPE const initialSize = array->size();
    EXPECT_EQ( array->capacity(), initialSize );

    INDEX_TYPE const defaultDimSize = array->size( 0 );
    array->reserve( 2 * initialSize );
    T const * const pointerAfterReserve = array->data();

    EXPECT_EQ( array->size(), initialSize );
    EXPECT_EQ( array->capacity(), 2 * initialSize );

    array->resize( 1.5 * defaultDimSize );
    fill( *array );

    EXPECT_EQ( array->capacity(), 2 * initialSize );
    EXPECT_EQ( array->data(), pointerAfterReserve );

    array->resize( 2 * defaultDimSize );
    fill( *array );

    EXPECT_EQ( array->capacity(), 2 * initialSize );
    EXPECT_EQ( array->data(), pointerAfterReserve );

    array->resize( 0.5 * defaultDimSize );
    fill( *array );

    EXPECT_EQ( array->capacity(), 2 * initialSize );
    EXPECT_EQ( array->data(), pointerAfterReserve );
  }

  static void clear()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();

    std::array< INDEX_TYPE, NDIM > oldSizes;
    for( int dim = 0; dim < NDIM; ++dim )
    { oldSizes[ dim ] = array->size( dim ); }

    INDEX_TYPE const oldSize = array->size();
    INDEX_TYPE const oldCapacity = array->capacity();
    T const * const oldPointer = array->data();

    array->clear();

    EXPECT_EQ( array->size(), 0 );
    EXPECT_EQ( array->capacity(), oldCapacity );
    EXPECT_EQ( array->data(), oldPointer );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      if( dim == 0 )
      {
        EXPECT_EQ( array->size( dim ), 0 );
      }
      else
      {
        EXPECT_EQ( array->size( dim ), oldSizes[ dim ] );
      }
    }

    array->resize( oldSizes[ 0 ] );

    EXPECT_EQ( array->size(), oldSize );
    EXPECT_EQ( array->capacity(), oldCapacity );
    EXPECT_EQ( array->data(), oldPointer );

    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( array->size( dim ), oldSizes[ dim ] );
    }

    fill( *array );
  }

  static void checkIndexing()
  {
    ARRAY array;

    std::array< INDEX_TYPE, NDIM > dimensions;
    for( int i = 0; i < NDIM; ++i )
    { dimensions[ i ] = randomInteger( 1, getMaxDimSize() ); }

    array.resize( NDIM, dimensions.data() );
    fill( array );

    RAJA::View< T, RAJA::Layout< NDIM > > view( array.data(),
                                                RAJA::make_permuted_layout( dimensions,
                                                                            RAJA::as_array< PERMUTATION >::get() ) );

    EXPECT_EQ( array.data(), getRAJAViewData( view ) );

    RAJA::Layout< NDIM > const & layout = getRAJAViewLayout( view );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( array.size( dim ), layout.sizes[ dim ] );
      EXPECT_EQ( array.strides()[ dim ], layout.strides[ dim ] );
    }

    compareToRAJAView( array, view );
  }

protected:

  static LVARRAY_HOST_DEVICE INDEX_TYPE
  getTestingLinearIndex( INDEX_TYPE const i )
  { return i; }

  static LVARRAY_HOST_DEVICE INDEX_TYPE
  getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j )
  {  return getMaxDimSize() * i + j; }

  static LVARRAY_HOST_DEVICE INDEX_TYPE
  getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const k )
  {
    INDEX_TYPE const maxDimSize = getMaxDimSize();
    return maxDimSize * maxDimSize * i + maxDimSize * j + k;
  }

  static LVARRAY_HOST_DEVICE INDEX_TYPE
  getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const k, INDEX_TYPE const l )
  {
    INDEX_TYPE const maxDimSize = getMaxDimSize();
    return maxDimSize * maxDimSize * maxDimSize * i + maxDimSize * maxDimSize * j + maxDimSize * k + l;
  }

  static void fill( ARRAY const & array )
  {
    forValuesInSliceWithIndices( array.toSlice(), [] ( T & value, auto const ... indices )
    {
      value = T( getTestingLinearIndex( indices ... ) );
    } );

    checkFill( array );
  }

  static void checkFill( ARRAY const & array )
  {
    forValuesInSliceWithIndices( array.toSliceConst(), [&array] ( T const & value, auto const ... indices )
    {
      EXPECT_EQ( value, T( getTestingLinearIndex( indices ... ) ) );
      EXPECT_EQ( &value, &array( indices ... ) );
      EXPECT_EQ( &value - array.data(), array.linearIndex( indices ... ) );
    } );

    // Check using the iterator interface
    INDEX_TYPE offset = 0;
    for( T const & val : array )
    {
      EXPECT_EQ( val, array.data()[ offset ] );
      ++offset;
    }
  }

  static void checkFill( ARRAY const & array,
                         std::array< INDEX_TYPE, NDIM > const & oldSizes,
                         bool const defaultInitialized,
                         T const defaultValue )
  {
    forValuesInSliceWithIndices( array.toSliceConst(),
                                 [&array, oldSizes, defaultInitialized, defaultValue] ( T const & value, auto const ... indices )
    {
      if( !indexing::invalidIndices( oldSizes.data(), indices ... ) )
      {
        EXPECT_EQ( value, T( getTestingLinearIndex( indices ... ) ) );
        EXPECT_EQ( &value, &array( indices ... ) );
      }
      else if( defaultInitialized )
      {
        EXPECT_EQ( value, defaultValue );
        EXPECT_EQ( &value, &array( indices ... ) );
      }
    } );
  }

  template< typename ARRAY_A, typename ARRAY_B >
  static void compare( ARRAY_A const & a, ARRAY_B const & b, bool const shallowCopy=false )
  {
    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( a.size( dim ), b.size( dim ) );
    }

    EXPECT_EQ( a.size(), b.size() );

    forValuesInSliceWithIndices( a.toSliceConst(), [&a, &b, shallowCopy]( T const & val, auto const ... indices )
    {
      EXPECT_EQ( val, a( indices ... ) );
      EXPECT_EQ( val, b( indices ... ) );

      if( shallowCopy )
      { EXPECT_EQ( &a( indices ... ), &b( indices ... ) ); }
      else
      { EXPECT_NE( &a( indices ... ), &b( indices ... ) ); }
    } );
  }

  static void checkResize( ARRAY const & array,
                           std::array< INDEX_TYPE, NDIM > const & oldSizes,
                           std::array< INDEX_TYPE, NDIM > const & newSizes,
                           bool const valuesPreserved,
                           bool const defaultInitialized,
                           T const defaultValue=T() )
  {
    INDEX_TYPE totalSize = 1;
    for( int dim = 0; dim < NDIM; ++dim )
    {
      EXPECT_EQ( array.size( dim ), newSizes[ dim ] );
      totalSize *= newSizes[ dim ];
    }

    EXPECT_EQ( array.size(), totalSize );
    EXPECT_GE( array.capacity(), totalSize );

    if( valuesPreserved )
    { checkFill( array, oldSizes, defaultInitialized, defaultValue ); }

    fill( array );
  }

  template< int DIM0 >
  static void resizeOneDimension( ARRAY & array )
  {
    std::array< INDEX_TYPE, NDIM > oldSizes;
    for( int dim = 0; dim < NDIM; ++dim )
    { oldSizes[ dim ] = array.size( dim ); }

    std::array< INDEX_TYPE, NDIM > newSizes = oldSizes;

    // Increase the size
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 3;
    array.template resizeDimension< DIM0 >( newSizes[ DIM0 ] );
    checkResize( array, oldSizes, newSizes, NDIM == 1, true );
    oldSizes = newSizes;

    // Shrink the size
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 5;
    array.template resizeDimension< DIM0 >( newSizes[ DIM0 ] );
    checkResize( array, oldSizes, newSizes, NDIM == 1, true );
    oldSizes = newSizes;

    // Return to the original size
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 2;
    array.template resizeDimension< DIM0 >( newSizes[ DIM0 ] );
    checkResize( array, oldSizes, newSizes, NDIM == 1, true );
  }

  template< int DIM0, int DIM1 >
  static void resizeTwoDimensions( ARRAY & array )
  {
    std::array< INDEX_TYPE, NDIM > oldSizes;
    for( int dim = 0; dim < NDIM; ++dim )
    { oldSizes[ dim ] = array.size( dim ); }

    std::array< INDEX_TYPE, NDIM > newSizes = oldSizes;

    // Increase the size of both dimension
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 3;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 2;
    array.template resizeDimension< DIM0, DIM1 >( newSizes[ DIM0 ], newSizes[ DIM1 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the size of both dimension
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 4;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 1;
    array.template resizeDimension< DIM0, DIM1 >( newSizes[ DIM0 ], newSizes[ DIM1 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Grow the first dimension and shrink the second.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 2;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 4;
    array.template resizeDimension< DIM0, DIM1 >( newSizes[ DIM0 ], newSizes[ DIM1 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the first dimension and grow the second.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 1;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 3;
    array.template resizeDimension< DIM0, DIM1 >( newSizes[ DIM0 ], newSizes[ DIM1 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;
  }

  template< int DIM0, int DIM1, int DIM2 >
  static void resizeThreeDimensions( ARRAY & array )
  {
    std::array< INDEX_TYPE, NDIM > oldSizes;
    for( int dim = 0; dim < NDIM; ++dim )
    { oldSizes[ dim ] = array.size( dim ); }

    std::array< INDEX_TYPE, NDIM > newSizes = oldSizes;

    // Increase the size of all dimension
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 3;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 1;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] + 2;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the size of all dimension
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 4;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 2;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] - 1;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the first, grow the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 1;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 3;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] + 2;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the second, grow the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 3;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 3;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] + 1;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Shrink the third, grow the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 3;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 4;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] - 4;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Grow the first, shrink the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] + 2;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 2;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] - 2;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Grow the second, shrink the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 4;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] + 3;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] - 1;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;

    // Grow the third, shrink the rest.
    newSizes[ DIM0 ] = oldSizes[ DIM0 ] - 2;
    newSizes[ DIM1 ] = oldSizes[ DIM1 ] - 4;
    newSizes[ DIM2 ] = oldSizes[ DIM2 ] + 3;
    array.template resizeDimension< DIM0, DIM1, DIM2 >( newSizes[ DIM0 ], newSizes[ DIM1 ], newSizes[ DIM2 ] );
    checkResize( array, oldSizes, newSizes, false, true );
    oldSizes = newSizes;
  }

  static void compareToRAJAView( ARRAY const & array,
                                 RAJA::View< T, RAJA::Layout< 1 > > const & view )
  {
    for( INDEX_TYPE i = 0; i < array.size( 0 ); ++i )
    {
      EXPECT_EQ( &view( i ), &array[ i ] );
      EXPECT_EQ( &view( i ), &array( i ) );
    }
  }

  static void compareToRAJAView( ARRAY const & array,
                                 RAJA::View< T, RAJA::Layout< 2 > > const & view )
  {
    for( INDEX_TYPE i = 0; i < array.size( 0 ); ++i )
    {
      for( INDEX_TYPE j = 0; j < array.size( 1 ); ++j )
      {
        EXPECT_EQ( &view( i, j ), &array[ i ][ j ] );
        EXPECT_EQ( &view( i, j ), &array( i, j ) );
      }
    }
  }

  static void compareToRAJAView( ARRAY const & array,
                                 RAJA::View< T, RAJA::Layout< 3 > > const & view )
  {
    for( INDEX_TYPE i = 0; i < array.size( 0 ); ++i )
    {
      for( INDEX_TYPE j = 0; j < array.size( 1 ); ++j )
      {
        for( INDEX_TYPE k = 0; k < array.size( 2 ); ++k )
        {
          EXPECT_EQ( &view( i, j, k ), &array[ i ][ j ][ k ] );
          EXPECT_EQ( &view( i, j, k ), &array( i, j, k ) );
        }
      }
    }
  }

  static void compareToRAJAView( ARRAY const & array,
                                 RAJA::View< T, RAJA::Layout< 4 > > const & view )
  {
    for( INDEX_TYPE i = 0; i < array.size( 0 ); ++i )
    {
      for( INDEX_TYPE j = 0; j < array.size( 1 ); ++j )
      {
        for( INDEX_TYPE k = 0; k < array.size( 2 ); ++k )
        {
          for( INDEX_TYPE l = 0; l < array.size( 3 ); ++l )
          {
            EXPECT_EQ( &view( i, j, k, l ), &array[ i ][ j ][ k ][ l ] );
            EXPECT_EQ( &view( i, j, k, l ), &array( i, j, k, l ) );
          }
        }
      }
    }
  }


  template< typename LAMBDA >
  static auto forwardArrayAsArgs( std::array< INDEX_TYPE, 1 > const & sizes, LAMBDA && lambda )
  { return lambda( sizes[ 0 ] ); }

  template< typename LAMBDA >
  static auto forwardArrayAsArgs( std::array< INDEX_TYPE, 2 > const & sizes, LAMBDA && lambda )
  { return lambda( sizes[ 0 ], sizes[ 1 ] ); }

  template< typename LAMBDA >
  static auto forwardArrayAsArgs( std::array< INDEX_TYPE, 3 > const & sizes, LAMBDA && lambda )
  { return lambda( sizes[ 0 ], sizes[ 1 ], sizes[ 2 ] ); }

  template< typename LAMBDA >
  static auto forwardArrayAsArgs( std::array< INDEX_TYPE, 4 > const & sizes, LAMBDA && lambda )
  { return lambda( sizes[ 0 ], sizes[ 1 ], sizes[ 2 ], sizes[ 3 ] ); }

  static LVARRAY_HOST_DEVICE INDEX_TYPE getMaxDimSize()
  {
    INDEX_TYPE const maxDimSizes[ 4 ] = { 1000, 50, 20, 10 };
    return maxDimSizes[ NDIM - 1 ];
  }
};

using ArrayTestTypes = ::testing::Types<
  Array< int, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_IKJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_JIK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_JKI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_KIJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< TestString, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >
  >;

TYPED_TEST_SUITE( ArrayTest, ArrayTestTypes, );

template< typename ARRAY >
class ArrayOfTrivialObjectsTest : public ArrayTest< ARRAY >
{
public:
  using ArrayTest< ARRAY >::NDIM;
  using ArrayTest< ARRAY >::sizedConstructor;

  static void resizeWithoutInitializationOrDestruction()
  {
    std::unique_ptr< ARRAY > array = sizedConstructor();

    std::array< INDEX_TYPE, NDIM > newSizes;
    std::array< INDEX_TYPE, NDIM > oldSizes;

    // Shrink the size of each dimension.
    for( int dim = 0; dim < NDIM; ++dim )
    {
      newSizes[ dim ] = dim + 5;
      oldSizes[ dim ] = array->size( dim );
    }
    forwardArrayAsArgs( newSizes, [&array]( auto const ... indices )
    {
      return array->resizeWithoutInitializationOrDestruction( indices ... );
    } );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, false );

    // Increase the size of every dimension.
    for( int dim = 0; dim < NDIM; ++dim )
    {
      newSizes[ dim ] = dim + 15;
      oldSizes[ dim ] = array->size( dim );
    }
    forwardArrayAsArgs( newSizes, [&array]( auto const ... indices )
    {
      return array->resizeWithoutInitializationOrDestruction( indices ... );
    } );
    checkResize( *array, oldSizes, newSizes, NDIM == 1, false );
  }

protected:
  using ArrayTest< ARRAY >::forwardArrayAsArgs;
  using ArrayTest< ARRAY >::checkResize;
};

using ArrayOfTrivialObjectsTestTypes = ::testing::Types<
  Array< int, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_IKJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_JIK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_JKI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_KIJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< int, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 2, RAJA::PERM_IJ, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 2, RAJA::PERM_JI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 3, RAJA::PERM_IJK, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 3, RAJA::PERM_KJI, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 4, RAJA::PERM_IJKL, INDEX_TYPE, DEFAULT_BUFFER >
  , Array< Tensor, 4, RAJA::PERM_LKJI, INDEX_TYPE, DEFAULT_BUFFER >
  >;

TYPED_TEST_SUITE( ArrayOfTrivialObjectsTest, ArrayOfTrivialObjectsTestTypes, );

} // namespace testing
} // namespace LvArray
