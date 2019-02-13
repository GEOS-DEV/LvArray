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

/*
 * helpers.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: settgast
 */

#ifndef SRC_SRC_HELPERS_HPP_
#define SRC_SRC_HELPERS_HPP_

#include "Logger.hpp"

namespace LvArray
{

/**
 * @struct this is a functor to calculate the total size of the array from the dimensions.
 */
template< int NDIM, typename INDEX_TYPE, int DIM=0 >
struct size_helper
{
  template< int INDEX=DIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if<INDEX!=NDIM-1, INDEX_TYPE>::type
  f( INDEX_TYPE const * const restrict dims )
  {
    return dims[INDEX] * size_helper<NDIM, INDEX_TYPE, INDEX+1>::f( dims );
  }

  template< int INDEX=DIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if<INDEX==NDIM-1, INDEX_TYPE>::type
  f( INDEX_TYPE const * const restrict dims )
  {
    return dims[INDEX];
  }

};

template< int NDIM, typename INDEX_TYPE >
struct stride_helper
{
  template< int N=0 >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if< N!=NDIM-1, INDEX_TYPE>::type
  evaluate( INDEX_TYPE const * const restrict dims )
  {
    return dims[1]*stride_helper< NDIM, INDEX_TYPE >::template evaluate<N+1>( dims+1 );
  }

  template< int N=0 >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if< N==NDIM-1, INDEX_TYPE>::type
  evaluate( INDEX_TYPE const * const restrict )
  {
    return 1;
  }
};

template< int NDIM, typename INDEX_TYPE, typename INDEX, typename... REMAINING_INDICES >
struct linearIndex_helper
{
  template< int DIM=0 >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if< DIM!=(NDIM-1), INDEX_TYPE>::type
  evaluate( INDEX_TYPE const * const restrict strides,
            INDEX index,
            REMAINING_INDICES... indices )
  {
    return index*strides[0]
           + linearIndex_helper< NDIM,
                                 INDEX_TYPE,
                                 REMAINING_INDICES...>::template evaluate<DIM+1>( strides+1,
                                                                                  indices... );
  }

  template< int DIM=0 >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if< DIM==(NDIM-1), INDEX_TYPE>::type
  evaluate( INDEX_TYPE const * const restrict strides,
            INDEX index )
  {
    return index;
  }



  template< int DIM=0 >
  LVARRAY_HOST_DEVICE inline  static
  typename std::enable_if< DIM!=(NDIM-1), void>::type
  check( INDEX_TYPE const * const restrict dims,
         INDEX index, REMAINING_INDICES... indices )
  {
    GEOS_ERROR_IF( index < 0 || index >= dims[0], "index=" << index << ", m_dims[" <<
                   DIM << "]=" << dims[0] );
    linearIndex_helper< NDIM,
                        INDEX_TYPE,
                        REMAINING_INDICES...>::template check<DIM+1>( dims + 1,
                                                                      indices... );
  }

  template< int DIM=0 >
  LVARRAY_HOST_DEVICE inline  static
  typename std::enable_if< DIM==(NDIM-1), void>::type
  check( INDEX_TYPE const * const restrict dims,
         INDEX index )
  {
    GEOS_ERROR_IF( index < 0 || index >= dims[0], "index=" << index << ", m_dims[" <<
                   DIM << "]=" << dims[0] );
  }

};

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


template <bool... B>
struct conjunction {};

template <bool Head, bool... Tail>
struct conjunction<Head, Tail...>
  : std::integral_constant<bool, Head && conjunction<Tail...>::value> {};

template <bool B>
struct conjunction<B> : std::integral_constant<bool, B> {};



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
  constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... )
  {
    m_dims[NDIM-1] = dim0;
    return 0;
  }
};



template< typename INDEX_TYPE, int NDIM, typename... DIMS>
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence<INDEX_TYPE> indices,
                                        DIMS... dims );

template< typename INDEX_TYPE, int NDIM, INDEX_TYPE INDEX0, INDEX_TYPE... INDICES, typename DIM0, typename... DIMS >
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence<INDEX_TYPE, INDEX0, INDICES...> indices,
                                        DIM0 dim0, DIMS... dims )
{
  m_dims[INDEX0] = dim0;
  dim_index_unpack<INDEX_TYPE, NDIM>( m_dims, std::integer_sequence<INDEX_TYPE, INDICES...>(), dims... );
}
template< typename INDEX_TYPE, int NDIM, typename... DIMS>
constexpr static void dim_index_unpack( INDEX_TYPE m_dims[NDIM],
                                        std::integer_sequence<INDEX_TYPE> indices,
                                        DIMS... dims )
{
  // terminates recursion trivially
}


template< typename T,
          int NDIM,
          typename INDEX_TYPE=std::ptrdiff_t,
          typename DATA_VECTOR_TYPE=ChaiVector<T> >
class Array;

template< typename T,
          int NDIM,
          typename INDEX_TYPE=std::ptrdiff_t,
          typename DATA_VECTOR_TYPE=ChaiVector<T> >
class ArrayView;

namespace detail
{


template<typename>
struct is_array : std::false_type {};


template< typename T,
          int NDIM,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
struct is_array< Array<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > > : std::true_type {};


template<typename T>
struct to_arrayView
{
  using type = T;
};


template< typename T,
          int NDIM,
          typename INDEX_TYPE >
struct to_arrayView< Array< T, NDIM, INDEX_TYPE > >
{
  using type = ArrayView< T, NDIM, INDEX_TYPE > const;
};

/* Note this will only work when using ChaiVector as the DATA_VECTOR_TYPE. */
template< typename T,
          int NDIM1,
          typename INDEX_TYPE1,
          int NDIM0,
          typename INDEX_TYPE0 >
struct to_arrayView< Array< Array< T, NDIM1, INDEX_TYPE1 >,
                            NDIM0,
                            INDEX_TYPE0 > > 
{
  using type = ArrayView< typename to_arrayView< Array< T, NDIM1, INDEX_TYPE1 > >::type,
                          NDIM0,
                          INDEX_TYPE0 > const;
};

template<typename T>
struct to_arrayViewConst
{
  using type = T;
};


template< typename T,
          int NDIM,
          typename INDEX_TYPE >
struct to_arrayViewConst< Array< T, NDIM, INDEX_TYPE > >
{
  using type = ArrayView< T const, NDIM, INDEX_TYPE > const;
};

/* Note this will only work when using ChaiVector as the DATA_VECTOR_TYPE. */
template< typename T,
          int NDIM1,
          typename INDEX_TYPE1,
          int NDIM0,
          typename INDEX_TYPE0 >
struct to_arrayViewConst< Array< Array< T, NDIM1, INDEX_TYPE1 >,
                            NDIM0,
                            INDEX_TYPE0 > > 
{
  using type = ArrayView< typename to_arrayViewConst< Array< T, NDIM1, INDEX_TYPE1 > >::type,
                          NDIM0,
                          INDEX_TYPE0 > const;
};

} /* namespace detail */

} /* namespace LvArray */

#endif /* SRC_SRC_HELPERS_HPP_ */
