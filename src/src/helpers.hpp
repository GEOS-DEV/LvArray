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
    return dims[INDEX] * size_helper<NDIM,INDEX_TYPE,INDEX+1>::f( dims );
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


namespace DefaultValueHelper
{
template< typename T, typename ENABLE=void >
struct Helper
{
  static constexpr bool has_default_value = false;
};

template< typename T >
struct Helper< T, typename std::enable_if< std::is_arithmetic<T>::value >::type >
{
  static constexpr bool has_default_value = true;
  using value_type = T;
  value_type value = 0;
};
}

template< typename T >
using DefaultValue = DefaultValueHelper::Helper<T>;


}



#endif /* SRC_SRC_HELPERS_HPP_ */
