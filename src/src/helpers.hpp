/*
 * helpers.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: settgast
 */

#ifndef SRC_SRC_HELPERS_HPP_
#define SRC_SRC_HELPERS_HPP_


namespace LvArray
{
/**
 * @struct this is a functor to calculate the total size of the array from the dimensions.
 */
template< int NDIM, typename INDEX_TYPE, int DIM=0 >
struct size_helper
{
  template< int INDEX=DIM >
  inline CONSTEXPRFUNC static
  typename std::enable_if<INDEX!=NDIM-1, INDEX_TYPE>::type
  f( INDEX_TYPE const * const restrict dims )
  {
    return dims[INDEX] * size_helper<NDIM, INDEX_TYPE, INDEX+1>::f( dims );
  }

  template< int INDEX=DIM >
  inline CONSTEXPRFUNC static
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

#if 0
/**
 * @struct This is a functor to calculate the linear index of a multidimensional array.
 */
template< int NDIM,
          int DIM,
          typename INDEX_TYPE,
          typename INDEX,
          typename... REMAINING_INDICES >
struct linearIndex_helper
{
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static INDEX_TYPE
  evaluate( INDEX_TYPE const * const restrict strides,
            INDEX index,
            REMAINING_INDICES... indices )
  {
    return index*strides[0]
           + linearIndex_helper<NDIM, DIM-1, INDEX_TYPE, REMAINING_INDICES...>::evaluate( strides+1,
                                                                                          indices... );
  }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static void
  check( INDEX_TYPE const * const restrict dims,
         INDEX index, REMAINING_INDICES... indices )
  {
    GEOS_ERROR_IF( index < 0 || index >= dims[0], "index=" << index << ", m_dims[" <<
                   (NDIM - DIM) << "]=" << dims[0] );
    linearIndex_helper<NDIM, DIM-1, INDEX_TYPE, REMAINING_INDICES...>::check( dims + 1, indices... );
  }

};

template< int NDIM,
          typename INDEX_TYPE,
          typename INDEX,
          typename... REMAINING_INDICES >
struct linearIndex_helper<NDIM, 1, INDEX_TYPE, INDEX, REMAINING_INDICES...>
{
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static INDEX_TYPE
  evaluate( INDEX_TYPE const * const restrict,
            INDEX index )
  {
    return index;
  }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static void
  check( INDEX_TYPE const * const restrict dims,
         INDEX index )
  {
    GEOS_ERROR_IF( index < 0 || index >= dims[0], "index=" << index << ", m_dims[" <<
                   (NDIM - 1) << "]=" << dims[0] );
  }
};

#else

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
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
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
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static
  typename std::enable_if< DIM==(NDIM-1), void>::type
  check( INDEX_TYPE const * const restrict dims,
         INDEX index )
  {
    GEOS_ERROR_IF( index < 0 || index >= dims[0], "index=" << index << ", m_dims[" <<
                   DIM << "]=" << dims[0] );
  }

};
#endif


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


template< typename T,
          int NDIM,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
class Array;

namespace detail
{
template<typename>
struct is_array : std::false_type {};

template< typename T,
          int NDIM,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
struct is_array< Array<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > > : std::true_type {};
}



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

}

#endif /* SRC_SRC_HELPERS_HPP_ */
