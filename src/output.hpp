/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file output.hpp
 * @brief Contains functions for outputting array objects.
 */

#pragma once

// Source includes
#include "Array.hpp"
#include "SortedArray.hpp"
#include "ArrayOfArrays.hpp"
#include "CRSMatrix.hpp"
#include "Macros.hpp"
#include "limits.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <string>
#include <iostream>

#if defined( LVARRAY_USE_CUDA )
  #include <cuda_fp16.h>
#endif

namespace LvArray
{

/**
 * @tparam T The type of the values in @p slice.
 * @tparam NDIM The number of dimensions of @p slice.
 * @tparam USD The unit stride dimension of @p slice.
 * @tparam INDEX_TYPE The integer used by @p slice.
 * @brief This function outputs the contents of an array slice to an output stream.
 * @param stream The output stream to write to.
 * @param slice The slice to output.
 * @return @p stream .
 */
// Sphinx start after Array stream IO
template< typename T, int NDIM, int USD, typename INDEX_TYPE >
std::ostream & operator<<( std::ostream & stream,
                           ::LvArray::ArraySlice< T, NDIM, USD, INDEX_TYPE > const slice )
{
  stream << "{ ";

  if( slice.size( 0 ) > 0 )
  {
    stream << slice[ 0 ];
  }

  for( INDEX_TYPE i = 1; i < slice.size( 0 ); ++i )
  {
    stream << ", " << slice[ i ];
  }

  stream << " }";
  return stream;
}
// Sphinx end before Array stream IO


/**
 * @tparam T The type of the values in @p view.
 * @tparam NDIM The number of dimensions of @p view.
 * @tparam USD The unit stride dimension of @p view.
 * @tparam INDEX_TYPE The integer used by @p view.
 * @tparam BUFFER_TYPE The buffer type used by @p view.
 * @brief This function outputs the contents of an ArrayView to an output stream.
 * @param stream The output stream to write to.
 * @param view The view to output.
 * @return @p stream .
 */
template< typename T,
          int NDIM,
          int USD,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
std::ostream & operator<<( std::ostream & stream,
                           ::LvArray::ArrayView< T, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const & view )
{ return stream << view.toSliceConst(); }

/**
 * @tparam T The type of the values in @p view.
 * @tparam INDEX_TYPE The integer used by @p view.
 * @brief This function outputs the contents of @p view to an output stream.
 * @param stream The output stream to write to.
 * @param view The SortedArrayView to output.
 * @return @p stream .
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::ostream & operator<< ( std::ostream & stream,
                            SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const & view )
{
  if( view.size() == 0 )
  {
    stream << "{}";
    return stream;
  }

  stream << "{ ";

  if( view.size() > 0 )
  {
    stream << view[ 0 ];
  }

  for( INDEX_TYPE i = 1; i < view.size(); ++i )
  {
    stream << ", " << view[ i ];
  }

  stream << " }";
  return stream;
}

/**
 * @tparam T The type of the values in @p array.
 * @tparam INDEX_TYPE The integer used by @p array.
 * @brief This function outputs the contents of @p array to an output stream.
 * @param stream The output stream to write to.
 * @param array The SortedArray to output.
 * @return @p stream .
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::ostream & operator<< ( std::ostream & stream,
                            SortedArray< T, INDEX_TYPE, BUFFER_TYPE > const & array )
{ return stream << array.toViewConst(); }

/**
 * @tparam T The type of the values in @p view.
 * @tparam INDEX_TYPE The integer used by @p view.
 * @brief This function outputs the contents of @p view to an output stream.
 * @param stream The output stream to write to.
 * @param view The ArrayOfArraysView to output.
 * @return @p stream .
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::ostream & operator<< ( std::ostream & stream,
                            ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE > const & view )
{
  stream << "{" << std::endl;

  for( INDEX_TYPE i = 0; i < view.size(); ++i )
  {
    stream << i << "\t{";
    for( INDEX_TYPE j = 0; j < view.sizeOfArray( i ); ++j )
    {
      stream << view( i, j ) << ", ";
    }

    stream << "}" << std::endl;
  }

  stream << "}" << std::endl;
  return stream;
}

/**
 * @tparam T The type of the values in @p array.
 * @tparam INDEX_TYPE The integer used by @p array.
 * @brief This function outputs the contents of @p array to an output stream.
 * @param stream The output stream to write to.
 * @param array The ArrayOfArrays to output.
 * @return @p stream .
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::ostream & operator<< ( std::ostream & stream,
                            ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > const & array )
{ return stream << array.toViewConst(); }

/**
 * @tparam T The type of the values in @p view.
 * @tparam INDEX_TYPE The integer used by @p view.
 * @brief This function outputs the contents of @p view to an output stream.
 * @param stream The output stream to write to.
 * @param view The ArrayOfArraysView to output.
 * @return @p stream .
 */
template< typename T, typename COL_TYPE, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::ostream & operator<< ( std::ostream & stream, CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const & view )
{
  stream << "{" << std::endl;

  for( INDEX_TYPE row = 0; row < view.numRows(); ++row )
  {
    stream << "row " << row << std::endl;
    stream << "\tcolumns: " << view.getColumns( row ) << std::endl;
    stream << "\tvalues: " << view.getEntries( row ) << std::endl;
  }

  stream << "}" << std::endl;
  return stream;
}


/**
 * @brief Print a CRSMatrixView in a format that can be easily xxdiff'ed on the console.
 * @tparam POLICY The policy for the kernel launch.
 * @tparam T The value type in @p view.
 * @tparam T The column type in @p view.
 * @tparam T The index type in @p view.
 * @tparam BUFFER_TYPE The type of buffer used by @p view.
 * @param view The matrix view object to print.
 */
template< typename POLICY, typename T, typename COL_TYPE, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
void print( CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const & view )
{
  INDEX_TYPE const numRows = view.numRows();

  printf( "numRows = %4lld \n", integerConversion< long long >( numRows ) );
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, 1 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const )
    {
      INDEX_TYPE const * const ncols = view.getSizes();
      INDEX_TYPE const * const row_indexes = view.getOffsets();
      COL_TYPE const * const cols = view.getColumns();
      T const * const values = view.getEntries();

      printf( "ncols       = { " );
      for( INDEX_TYPE i = 0; i < numRows; ++i )
      {
        printf( "%4lld, ", integerConversion< long long >( ncols[ i ] ) );
      }
      printf( " }\n" );

      printf( "row_indexes = { " );
      for( INDEX_TYPE i = 0; i < numRows + 1; ++i )
      {
        printf( "%4lld, ", integerConversion< long long >( row_indexes[ i ] ) );
      }
      printf( " }\n" );

      printf( "row      col      value \n" );
      printf( "----  --------- --------- \n" );
      for( INDEX_TYPE i = 0; i < numRows; ++i )
      {
        printf( "%4lld\n", integerConversion< long long >( ncols[ i ] ) );
        for( INDEX_TYPE j = 0; j < ncols[ i ]; ++j )
        {
          printf( "%4lld %9lld %9.2g\n",
                  integerConversion< long long >( i ),
                  integerConversion< long long >( cols[ row_indexes[ i ] + j ] ),
                  double( values[ row_indexes[ i ] + j ] ) );
        }
      }
    } );

  std::cout << std::endl;
}

/**
 * @brief Output a c-array to a stream.
 * @tparam T The type contained in the array.
 * @tparam N The size of the array.
 * @param stream The output stream to write to.
 * @param array The c-array to output.
 * @return @p stream.
 */
template< typename T, int N >
std::enable_if_t< !std::is_same< T, char >::value, std::ostream & >
operator<< ( std::ostream & stream, T const ( &array )[ N ] )
{
  stream << "{ " << array[ 0 ];
  for( int i = 1; i < N; ++i )
  {
    stream << ", " << array[ i ];
  }
  stream << " }";
  return stream;
}

/**
 * @brief Output a 2D c-array to a stream.
 * @tparam T The type contained in the array.
 * @tparam M The size of the first dimension.
 * @tparam N The size of the second dimension.
 * @param stream The output stream to write to.
 * @param array The 2D c-array to output.
 * @return @p stream.
 */
template< typename T, int M, int N >
std::ostream & operator<< ( std::ostream & stream, T const ( &array )[ M ][ N ] )
{
  stream << "{ " << array[ 0 ];
  for( int i = 1; i < M; ++i )
  {
    stream << ", " << array[ i ];
  }
  stream << " }";
  return stream;
}

} // namespace LvArray

#if defined( LVARRAY_USE_CUDA )

/**
 * @brief Output a @c __half to a stream.
 * @param stream The output stream to write to.
 * @param x The value to ouput.
 * @return @p stream.
 */
inline std::ostream & operator<<( std::ostream & stream, __half const x )
{ return stream << float( x ); }

/**
 * @brief Output a @c __half2 to a stream.
 * @param stream The output stream to write to.
 * @param x The value to ouput.
 * @return @p stream.
 */
inline std::ostream & operator<<( std::ostream & stream, __half2 const x )
{
  float2 const c = __half22float2( x );
  return stream << c.x << ", " << c.y;
}

#endif
