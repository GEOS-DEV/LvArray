/* ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef ARRAY_VIEW_HPP_
#define ARRAY_VIEW_HPP_
#include <vector>
#include <iostream>
#include <utility>
#define ARRAY_BOUNDS_CHECK 1

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif

#include <array>
#include <vector>

namespace multidimensionalArray
{

//template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
//class ManagedArray;


/**
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector,
 * NDIM=2->Matrix, etc. )
 * This class serves as a multidimensional array interface for a chunk of
 * memory. This is a lightweight
 * class that contains only pointers, on integer data member to hold the stride,
 * and another instantiation of type
 * ArrayAccesssor<T,NDIM-1> to represent the sub-array that is passed back upon
 * use of operator[]. Pointers to the
 * data and length of each dimension passed into the constructor, thus the class
 * does not own the data itself, nor does
 * it own the array that defines the shape of the data.
 */
template< typename T, int NDIM, typename INDEX_TYPE >
class ArrayView
{
public:

  /// deleted default constructor
  ArrayView() = delete;

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array
   * has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and
   * calculates stride.
   */
  //inline constexpr explicit constexpr
  inline explicit
  ArrayView( T * const restrict inputData,
             INDEX_TYPE const * const restrict inputDimensions,
             INDEX_TYPE const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputDimensions),
    m_strides(inputStrides)
  {}



  /**
   * User Defined Conversion operator to move from an ArrayView<T> to ArrayView<T const>
   */
//  template< bool JUNK = std::is_const<T>::value,
//            typename std::enable_if<!JUNK, int>::type = 0>
//  template< typename U=T, typename std::enable_if< !std::is_const<U>::value, int >::type = 0 >
//  template< typename U = T, typename = typename std::enable_if< !std::is_const<U>::value >::type >
//  operator ArrayView<T const,NDIM,INDEX_TYPE>() const
  template< typename U = T >
  operator typename std::enable_if< !std::is_const<U>::value ,ArrayView<T const,NDIM,INDEX_TYPE> >::type () const
  {
    return ArrayView<T const,NDIM,INDEX_TYPE>( const_cast<T const *>(m_data),
                                               m_dims,
                                               m_strides );
  }

  /**
   * User Defined Conversion operator to move from an ArrayView<T> to ArrayView<T const>
   */
  template< int U = NDIM,
            typename std::enable_if<U==1, int>::type = 0>
  inline
  operator T *() const
  {
    return m_data;
  }


//  template< typename T_CC = typename std::enable_if< !std::is_const<T>::value, T >::type >
//  ArrayView( ArrayView<T_CC,NDIM,INDEX_TYPE> const & source ):
//    m_data(const_cast<T const *>(source.m_data) ),
//    m_dims(source.m_dims),
//    m_strides(source.m_strides)
//  {
//  }


//  inline constexpr  ArrayView( ManagedArray<T,NDIM,INDEX_TYPE> const &
// ManagedArray ):
//    m_data(ManagedArray.m_data),
//    m_dims(ManagedArray.m_dims),
//    m_strides(ManagedArray.m_strides)
//  {}

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array
   * has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and
   * calculates stride.
   */
//  inline constexpr explicit  ArrayView( T * const restrict inputData ):
//    m_data(    inputData + maxDim() ),
//    m_dims( reinterpret_cast<INDEX_TYPE*>( inputData ) + 1 )
//    m_strides()
//  {}
//

  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type
   * ArrayView<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the
   * location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the
   * beginning of the data associated with its
   * sub-array.
   */
#if ARRAY_BOUNDS_CHECK == 1
  template< int U=NDIM >
  //inline constexpr typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  inline typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  //inline constexpr typename std::enable_if< U==1, T const & >::type
  inline typename std::enable_if< U==1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return m_data[ index ];
  }
#else
  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U >= 3, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  inline typename std::enable_if< U >= 3, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  //inline constexpr typename std::enable_if< U==2, T const * restrict >::type
  inline typename std::enable_if< U==2, T const * restrict >::type
  operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U==1, T const & >::type
  inline typename std::enable_if< U==1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    return m_data[ index ];
  }

#endif



#if ARRAY_BOUNDS_CHECK == 1
  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  inline typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U==1, T & >::type
  inline typename std::enable_if< U==1, T & >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return m_data[ index ];
  }
#else
  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U>=3, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  inline typename std::enable_if< U>=3, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  //inline constexpr  typename std::enable_if< U==2, T * restrict >::type
  inline typename std::enable_if< U==2, T * restrict >::type
  operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  //inline constexpr typename std::enable_if< U==1, T & >::type
  inline typename std::enable_if< U==1, T & >::type
  operator[](INDEX_TYPE const index)
  {
    return m_data[ index ];
  }

#endif

  template< typename... INDICES >
  //inline constexpr T & operator()( INDICES... indices ) const
  inline T & operator()( INDICES... indices ) const
  {
    return m_data[ linearIndex(indices...) ];
  }

  template< typename... INDICES >
  //inline constexpr  INDEX_TYPE linearIndex( INDICES... indices ) const
  inline INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    return index_helper<NDIM,INDICES...>::f(m_strides,indices...);
  }

  T * data() {return m_data;}
  T const * data() const {return m_data;}

  INDEX_TYPE size( int dim = 0 ) const
  {
    return m_dims[dim];
  }

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array
  // dimension
  INDEX_TYPE const * const restrict m_dims;

  INDEX_TYPE const * const restrict m_strides;


  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_helper
  {
    //inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
    inline static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                          INDEX index,
                                          REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_helper<1,INDEX,REMAINING_INDICES...>
  {
    //inline constexpr  static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
    inline static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
                                INDEX index,
                                REMAINING_INDICES... indices )
    {
      return index;
    }
  };

};




#if ARRAY_BOUNDS_CHECK==1
template< typename T, typename INDEX_TYPE >
using arrayView1d = ArrayView<T,1,INDEX_TYPE>;

#else

template< typename T, typename INDEX_TYPE = int>
using arrayView1d = T *;

#endif


} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
