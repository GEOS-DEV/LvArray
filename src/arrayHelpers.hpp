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
 * @file arrayHelpers.hpp
 */

#ifndef SRC_SRC_ARRAY_HELPERS_HPP_
#define SRC_SRC_ARRAY_HELPERS_HPP_

// Source includes
#include "Macros.hpp"
#include "Permutation.hpp"
#include "ChaiBuffer.hpp"
#include "NewChaiBuffer.hpp"
#include "templateHelpers.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

namespace LvArray
{

/**
 *
 */
template< std::ptrdiff_t i, std::ptrdiff_t exclude_i >
struct ConditionalMultiply
{

  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr A multiply( A const a, B const b )
  {
    // regular product term
    return a * b;
  }
};

template< std::ptrdiff_t i >
struct ConditionalMultiply< i, i >
{
  // Use a reference here for B so that you can do multiply( 5, *nullptr ), which is use by the ArrayOfArray classes.
  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr A multiply( A const a, B const & LVARRAY_UNUSED_ARG( b ) )
  {
    // assume b == 1
    return a;
  }
};

template< int SIZE, typename T >
LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
typename std::enable_if< (SIZE == 1), T >::type
multiplyAll( T const * const LVARRAY_RESTRICT values )
{ return values[ 0 ]; }

template< int SIZE, typename T >
LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
typename std::enable_if< (SIZE > 1), T >::type
multiplyAll( T const * const LVARRAY_RESTRICT values )
{ return values[ 0 ] * multiplyAll< SIZE - 1 >( values + 1 ); }

template< int USD, typename INDEX_TYPE, typename INDEX >
LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
INDEX_TYPE getLinearIndex( INDEX_TYPE const * const LVARRAY_RESTRICT strides, INDEX const index )
{
  return ConditionalMultiply< 0, USD >::multiply( index, strides[ 0 ] );
}

template< int USD, typename INDEX_TYPE, typename INDEX, typename ... REMAINING_INDICES >
LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
INDEX_TYPE getLinearIndex( INDEX_TYPE const * const LVARRAY_RESTRICT strides, INDEX const index, REMAINING_INDICES... indices )
{
  return ConditionalMultiply< 0, USD >::multiply( index, strides[ 0 ] ) +
         getLinearIndex< USD - 1, INDEX_TYPE, REMAINING_INDICES... >( strides + 1, indices ... );
}

template< typename INDEX_TYPE, typename INDEX, typename ... INDICES >
std::string printDimsAndIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDEX const index, INDICES... indices )
{
  constexpr int NDIM = sizeof ... (INDICES) + 1;
  std::ostringstream oss;
  oss << "dimensions = { " << dims[ 0 ];
  for( int i = 1; i < NDIM; ++i )
  {
    oss << ", " << dims[ i ];
  }
  oss << " } indices = { " << index;

  using expander = int[];
  (void) expander{ 0, ( void (oss << ", " << indices ), 0 )... };
  oss << " }";

  return oss.str();
}


template< typename none = void >
LVARRAY_HOST_DEVICE inline constexpr
bool invalidIndices( void const * const LVARRAY_RESTRICT LVARRAY_UNUSED_ARG( dims ) )
{ return false; }


template< typename INDEX_TYPE, typename INDEX, typename ... REMAINING_INDICES >
LVARRAY_HOST_DEVICE inline constexpr
bool invalidIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDEX const index, REMAINING_INDICES... indices )
{ return index < 0 || index >= dims[ 0 ] || invalidIndices( dims + 1, indices ... ); }


template< typename INDEX_TYPE, typename ... INDICES >
LVARRAY_HOST_DEVICE inline
void checkIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDICES... indices )
{ LVARRAY_ERROR_IF( invalidIndices( dims, indices ... ), "Invalid indices. " << printDimsAndIndices( dims, indices ... ) ); }


template< typename T >
struct is_integer
{
  constexpr static bool value = std::is_same< T, int >::value ||
                                std::is_same< T, unsigned int >::value ||
                                std::is_same< T, long int >::value ||
                                std::is_same< T, unsigned long int >::value ||
                                std::is_same< T, long long int >::value ||
                                std::is_same< T, unsigned long long int >::value;
};

template< typename INDEX_TYPE, typename CANDIDATE_INDEX_TYPE >
struct is_valid_indexType
{
  constexpr static bool value = std::is_same< CANDIDATE_INDEX_TYPE, INDEX_TYPE >::value ||
                                ( is_integer< CANDIDATE_INDEX_TYPE >::value &&
                                  ( sizeof(CANDIDATE_INDEX_TYPE)<=sizeof(INDEX_TYPE) ) );
};

template< typename INDEX_TYPE, typename DIM0, typename ... DIMS >
struct check_dim_type
{
  constexpr static bool value =  is_valid_indexType< INDEX_TYPE, DIM0 >::value && check_dim_type< INDEX_TYPE, DIMS... >::value;
};

template< typename INDEX_TYPE, typename DIM0 >
struct check_dim_type< INDEX_TYPE, DIM0 >
{
  constexpr static bool value = is_valid_indexType< INDEX_TYPE, DIM0 >::value;
};



template< typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0, INDEX_TYPE... INDICES >
struct check_dim_indices
{
  constexpr static bool value = (INDEX0 >= 0) && (INDEX0 < NDIM) && check_dim_indices< INDEX_TYPE, NDIM, INDICES... >::value;
};

template< typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0 >
struct check_dim_indices< INDEX_TYPE, NDIM, INDEX0 >
{
  constexpr static bool value = (INDEX0 >= 0) && (INDEX0 < NDIM);
};

template< typename INDEX_TYPE, typename ... DIMS >
void dimUnpack( INDEX_TYPE * const dims, DIMS... newDims )
{
  int curDim = 0;
  for_each_arg( [&]( INDEX_TYPE const size )
  {
    dims[ curDim ] = size;
    curDim += 1;
  }, newDims ... );
}


template< typename INDEX_TYPE, int NDIM, typename ... DIMS >
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence< INDEX_TYPE > indices,
                                        DIMS... dims );

template< typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0, INDEX_TYPE... INDICES, typename DIM0, typename ... DIMS >
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence< INDEX_TYPE, INDEX0, INDICES... > LVARRAY_UNUSED_ARG( indices ),
                                        DIM0 dim0, DIMS... dims )
{
  m_dims[INDEX0] = dim0;
  dim_index_unpack< INDEX_TYPE, NDIM >( m_dims, std::integer_sequence< INDEX_TYPE, INDICES... >(), dims ... );
}

template< typename INDEX_TYPE, int NDIM, typename ... DIMS >
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE LVARRAY_UNUSED_ARG( m_dims )[NDIM],
                                        std::integer_sequence< INDEX_TYPE > LVARRAY_UNUSED_ARG( indices ),
                                        DIMS... LVARRAY_UNUSED_ARG( dims ) )
{
  // terminates recursion trivially
}

template< typename T,
          int NDIM,
          typename PERMUTATION=camp::make_idx_seq_t< NDIM >,
          typename INDEX_TYPE=std::ptrdiff_t,
          template< typename > class DATA_VECTOR_TYPE=NewChaiBuffer >
class Array;

template< typename T,
          int NDIM,
          int USD=NDIM-1,
          typename INDEX_TYPE=std::ptrdiff_t,
          template< typename > class DATA_VECTOR_TYPE=NewChaiBuffer >
class ArrayView;


template< typename >
constexpr bool isArray = false;


template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class DATA_VECTOR_TYPE >
constexpr bool isArray< Array< T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > > = true;


template< typename T >
struct AsView
{
  using type = T;
};

template< typename T,
          int NDIM,
          int USD,
          typename INDEX_TYPE,
          template< typename > class DATA_VECTOR_TYPE >
struct AsView< ArrayView< T, NDIM, USD, INDEX_TYPE, DATA_VECTOR_TYPE > >
{
  using type = ArrayView< typename AsView< T >::type,
                          NDIM,
                          USD,
                          INDEX_TYPE,
                          DATA_VECTOR_TYPE > const;
};

template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class DATA_VECTOR_TYPE >
struct AsView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > >
{
  using type = ArrayView< typename AsView< T >::type,
  NDIM,
  getStrideOneDimension( PERMUTATION {} ),
  INDEX_TYPE,
  DATA_VECTOR_TYPE > const;
};

template< typename T >
struct AsConstView
{
  using type = T const;
};

template< typename T,
          int NDIM,
          int USD,
          typename INDEX_TYPE,
          template< typename > class DATA_VECTOR_TYPE >
struct AsConstView< ArrayView< T, NDIM, USD, INDEX_TYPE, DATA_VECTOR_TYPE > >
{
  using type = ArrayView< typename AsConstView< T >::type const,
                          NDIM,
                          USD,
                          INDEX_TYPE,
                          DATA_VECTOR_TYPE > const;
};

template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class DATA_VECTOR_TYPE >
struct AsConstView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > >
{
  using type = ArrayView< typename AsConstView< T >::type const,
  NDIM,
  getStrideOneDimension( PERMUTATION {} ),
  INDEX_TYPE,
  DATA_VECTOR_TYPE > const;
};

} /* namespace LvArray */

#endif /* SRC_SRC_ARRAY_HELPERS_HPP_ */
