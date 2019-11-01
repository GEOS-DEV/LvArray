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
 * @file helpers.hpp
 */

#ifndef SRC_SRC_HELPERS_HPP_
#define SRC_SRC_HELPERS_HPP_

#include "Permutation.hpp"
#include "Logger.hpp"
#include "ChaiVector.hpp"

#include <camp/camp.hpp>

namespace LvArray
{

/**
 * 
 */
template< int i, int exclude_i >
struct ConditionalMultiply
{

  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr A multiply( A const a, B const b )
  {
    // regular product term
    return a * b;
  }
};

template< int i >
struct ConditionalMultiply< i, i >
{
  // Use a reference here for B so that you can do multiply( 5, *nullptr ), which is use by the ArrayOfArray classes.
  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr A multiply( A const a, B const & CXX_UTILS_UNUSED_ARG( b ) )
  {
    // assume b == 1
    return a;
  }
};

template< int SIZE, typename T >
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if< (SIZE == 1), T >::type
multiplyAll( T const * const restrict values )
{ return values[ 0 ]; }

template< int SIZE, typename T >
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if< (SIZE > 1), T >::type
multiplyAll( T const * const restrict values )
{ return values[ 0 ] * multiplyAll< SIZE - 1 >( values + 1 ); }

template< int UNIT_STRIDE_DIM, typename INDEX_TYPE, typename none = void>
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
INDEX_TYPE getLinearIndex( void const * const restrict CXX_UTILS_UNUSED_ARG( strides ) )
{ return 0; }

template< int UNIT_STRIDE_DIM, typename INDEX_TYPE, typename INDEX, typename... REMAINING_INDICES >
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
INDEX_TYPE getLinearIndex( INDEX_TYPE const * const restrict strides, INDEX const index, REMAINING_INDICES... indices )
{
  return ConditionalMultiply< 0, UNIT_STRIDE_DIM >::multiply( index, strides[ 0 ] ) +
         getLinearIndex< UNIT_STRIDE_DIM - 1, INDEX_TYPE, REMAINING_INDICES...>( strides + 1, indices... );
}


template< typename INDEX_TYPE, typename INDEX, typename... INDICES >
std::string printDimsAndIndices( INDEX_TYPE const * const restrict dims, INDEX const index, INDICES... indices )
{
  constexpr int NDIM = sizeof ... (INDICES) + 1;
  std::ostringstream oss;
  oss << "dimensions = { " << dims[ 0 ];
  for ( int i = 1; i < NDIM; ++i )
  {
    oss << ", " << dims[ i ];
  }
  oss << " } indices = { " << index;
  
  using expander = int[];
  (void) expander{ 0, ( void (oss << ", " << indices ), 0 )... };
  oss << "}";

  return oss.str();
}


template< typename none = void>
LVARRAY_HOST_DEVICE inline constexpr
bool invalidIndices( void const * const restrict CXX_UTILS_UNUSED_ARG( dims ) )
{ return false; }


template< typename INDEX_TYPE, typename INDEX, typename... REMAINING_INDICES >
LVARRAY_HOST_DEVICE inline constexpr
bool invalidIndices( INDEX_TYPE const * const restrict dims, INDEX const index, REMAINING_INDICES... indices )
{ return index < 0 || index >= dims[ 0 ] || invalidIndices( dims + 1, indices... ); }


template< typename INDEX_TYPE, typename... INDICES >
LVARRAY_HOST_DEVICE inline
void checkIndices( INDEX_TYPE const * const restrict dims, INDICES... indices )
{ GEOS_ERROR_IF( invalidIndices( dims, indices... ), "Invalid indices. " << printDimsAndIndices( dims, indices... ) ); }


template< typename T >
struct is_integer
{
  constexpr static bool value = std::is_same<T, int>::value ||
                                std::is_same<T, unsigned int>::value ||
                                std::is_same<T, long int>::value ||
                                std::is_same<T, unsigned long int>::value ||
                                std::is_same<T, long long int>::value ||
                                std::is_same<T, unsigned long long int>::value;
};

template< typename INDEX_TYPE, typename CANDIDATE_INDEX_TYPE >
struct is_valid_indexType
{
  constexpr static bool value = std::is_same<CANDIDATE_INDEX_TYPE, INDEX_TYPE>::value ||
                                ( is_integer<CANDIDATE_INDEX_TYPE>::value &&
                                  ( sizeof(CANDIDATE_INDEX_TYPE)<=sizeof(INDEX_TYPE) ) );
};

template< typename INDEX_TYPE, typename DIM0, typename... DIMS >
struct check_dim_type
{
  constexpr static bool value =  is_valid_indexType<INDEX_TYPE, DIM0>::value && check_dim_type<INDEX_TYPE, DIMS...>::value;
};

template< typename INDEX_TYPE, typename DIM0 >
struct check_dim_type<INDEX_TYPE, DIM0>
{
  constexpr static bool value = is_valid_indexType<INDEX_TYPE, DIM0>::value;
};



template<  typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0, INDEX_TYPE... INDICES >
struct check_dim_indices
{
  constexpr static bool value = (INDEX0 >= 0) && (INDEX0 < NDIM) && check_dim_indices<INDEX_TYPE, NDIM, INDICES...>::value;
};

template<  typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0 >
struct check_dim_indices<INDEX_TYPE, NDIM, INDEX0>
{
  constexpr static bool value = (INDEX0 >= 0) && (INDEX0 < NDIM);
};

template<  typename INDEX_TYPE, int NDIM, int COUNTER, typename DIM0, typename... DIMS >
struct dim_unpack
{
  LVARRAY_HOST_DEVICE
  constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
  {
    m_dims[NDIM-COUNTER] = dim0;
    dim_unpack<  INDEX_TYPE, NDIM, COUNTER-1, DIMS...>::f( m_dims, dims... );
    return 0;
  }
};

template< typename INDEX_TYPE, int NDIM, typename DIM0, typename... DIMS >
struct dim_unpack<INDEX_TYPE, NDIM, 1, DIM0, DIMS...>
{
  LVARRAY_HOST_DEVICE
  constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... )
  {
    m_dims[NDIM-1] = dim0;
    return 0;
  }
};



template< typename INDEX_TYPE, int NDIM, typename... DIMS>
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence<INDEX_TYPE> indices,
                                        DIMS... dims );

template< typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0, INDEX_TYPE... INDICES, typename DIM0, typename... DIMS >
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence<INDEX_TYPE, INDEX0, INDICES...> CXX_UTILS_UNUSED_ARG( indices ),
                                        DIM0 dim0, DIMS... dims )
{
  m_dims[INDEX0] = dim0;
  dim_index_unpack<INDEX_TYPE, NDIM>( m_dims, std::integer_sequence<INDEX_TYPE, INDICES...>(), dims... );
}

template< typename INDEX_TYPE, int NDIM, typename... DIMS>
LVARRAY_HOST_DEVICE
constexpr static void dim_index_unpack( INDEX_TYPE CXX_UTILS_UNUSED_ARG( m_dims )[NDIM],
                                        std::integer_sequence<INDEX_TYPE> CXX_UTILS_UNUSED_ARG( indices ),
                                        DIMS... CXX_UTILS_UNUSED_ARG( dims ) )
{
  // terminates recursion trivially
}


template< typename T,
          int NDIM,
          typename PERMUTATION=camp::make_idx_seq_t<NDIM>,
          typename INDEX_TYPE=std::ptrdiff_t,
          typename DATA_VECTOR_TYPE=ChaiVector<T> >
class Array;

template< typename T,
          int NDIM,
          int UNIT_STRIDE_DIM=NDIM-1,
          typename INDEX_TYPE=std::ptrdiff_t,
          typename DATA_VECTOR_TYPE=ChaiVector<T> >
class ArrayView;

namespace detail
{


template<typename>
struct is_array : std::false_type {};


template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
struct is_array< Array<T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > > : std::true_type {};


template<typename T>
struct to_arrayView
{
  using type = T;
};


template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE >
struct to_arrayView< Array< T, NDIM, PERMUTATION, INDEX_TYPE > >
{
  using type = ArrayView< T, NDIM, getStrideOneDimension( PERMUTATION {} ), INDEX_TYPE > const;
};

/* Note this will only work when using ChaiVector as the DATA_VECTOR_TYPE. */
template< typename T,
          int NDIM1,
          typename PERMUTATION1,
          typename INDEX_TYPE1,
          int NDIM0,
          typename PERMUTATION0,
          typename INDEX_TYPE0 >
struct to_arrayView< Array< Array< T, NDIM1, PERMUTATION1, INDEX_TYPE1 >,
                            NDIM0,
                            PERMUTATION0,
                            INDEX_TYPE0 > >
{
  using type = ArrayView< typename to_arrayView< Array< T, NDIM1, PERMUTATION1, INDEX_TYPE1 > >::type,
                          NDIM0,
                          getStrideOneDimension( PERMUTATION0 {} ),
                          INDEX_TYPE0 > const;
};

template<typename T>
struct to_arrayViewConst
{
  using type = T;
};


template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE >
struct to_arrayViewConst< Array< T, NDIM, PERMUTATION, INDEX_TYPE > >
{
  using type = ArrayView< typename to_arrayViewConst<T>::type const,
                          NDIM,
                          getStrideOneDimension( PERMUTATION {} ),
                          INDEX_TYPE > const;
};

template< typename T,
          int NDIM,
          int UNIT_STRIDE_DIM,
          typename INDEX_TYPE >
struct to_arrayViewConst< ArrayView< T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE > >
{
  using type = ArrayView< typename to_arrayViewConst<T>::type const,
                          NDIM,
                          UNIT_STRIDE_DIM,
                          INDEX_TYPE > const;
};

} /* namespace detail */

} /* namespace LvArray */

#endif /* SRC_SRC_HELPERS_HPP_ */
