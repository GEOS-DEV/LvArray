/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#define SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#include<vector>
#include<iostream>
#include<utility>
#define ARRAY_BOUNDS_CHECK 0

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif

#include <array>
#include <vector>

typedef std::int_fast32_t integer_t;

namespace multidimensionalArray
{




template < size_t NDIM, size_t INDEX, typename INDEX_TYPE >
struct calculate_stride
{
  inline static constexpr INDEX_TYPE f(INDEX_TYPE const * const restrict dims )
  {
    return dims[INDEX] * calculate_stride<NDIM, INDEX+1, INDEX_TYPE>::f(dims);
  }
};



template < size_t NDIM, typename INDEX_TYPE >
struct calculate_stride< NDIM, NDIM ,INDEX_TYPE>
{
  inline static constexpr INDEX_TYPE f(INDEX_TYPE const * const restrict dims )
  {
    return 1;
  }
};




template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
class ArrayView;

template< typename T, int NDIM, typename INDEX_TYPE, typename INDEX_SEQUENCE >
class ManagedArray_impl {};



template< typename T, int NDIM, typename INDEX_TYPE, size_t... INDEX_SEQUENCE >
class ManagedArray_impl<T, NDIM, INDEX_TYPE, std::index_sequence<INDEX_SEQUENCE...> >
{
public:

  using Index_Sequence = std::make_index_sequence<NDIM>;
  /// deleted default constructor
//  ManagedArray_impl() = delete;

  template< typename... DIMS >
  inline explicit constexpr ManagedArray_impl( DIMS... dims ):
    dataVector(),
    m_data(),
    m_dims{dims...},
    m_strides{(calculate_stride<NDIM,INDEX_SEQUENCE+1,INDEX_TYPE>::f(m_dims))...}
  {

    INDEX_TYPE length = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      length *= m_dims[a];
    }

    dataVector.resize( length );
    m_data = dataVector.data();
  }

//  inline constexpr ArrayView<T,NDIM> View() const
//  {
//    return ArrayView<T,NDIM>(*this);
//  }
//
//  inline constexpr ArrayView<T,NDIM> View()
//  {
//    return ArrayView<T,NDIM>(*this);
//  }



//  inline explicit constexpr ArrayView( T * const restrict inputData ):
//    m_data(    inputData + maxDim() ),
//    m_dims( reinterpret_cast<INDEX_TYPE*>( inputData ) + 1 )
//    m_strides()
//  {}
//
  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArrayView<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the beginning of the data associated with its
   * sub-array.
   */

#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1> const operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U!=2 , ArrayView<T,NDIM-1> const >::type  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T const * restrict >::type operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }
#endif



#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1> operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U!=2 , ArrayView<T,NDIM-1> >::type  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T * restrict >::type operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }
#endif





  template< typename...DIMS >
  inline constexpr T & operator()( DIMS... dims ) const
  {
    return m_data[ index(dims...) ];
  }

  template< typename... DIMS >
  inline constexpr INDEX_TYPE index( DIMS... dims ) const
  {
    return index_helper<NDIM,DIMS...>::f(m_strides,dims...);
  }


//private:
  std::vector<T> dataVector;

  /// pointer to beginning of data for this array, or sub-array.
  T * restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const m_dims[NDIM];

  INDEX_TYPE const m_strides[NDIM];



  template< int DIM, typename INDEX, typename...REMAINING_INDICES >
  struct index_helper
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                         INDEX index,
                                         REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename...REMAINING_INDICES >
  struct index_helper<1,INDEX,REMAINING_INDICES...>
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
                                         INDEX index,
                                         REMAINING_INDICES... indices )
    {
      return index;
    }
  };

};


template < typename T, int NDIM , typename INDEX_TYPE = std::int_fast32_t >
using ManagedArray = ManagedArray_impl<T, NDIM, INDEX_TYPE, std::make_index_sequence<NDIM> >;





/**
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * This class serves as a multidimensional array interface for a chunk of memory. This is a lightweight
 * class that contains only pointers, on integer data member to hold the stride, and another instantiation of type
 * ArrayAccesssor<T,NDIM-1> to represent the sub-array that is passed back upon use of operator[]. Pointers to the
 * data and length of each dimension passed into the constructor, thus the class does not own the data itself, nor does
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
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and calculates stride.
   */
  inline explicit constexpr ArrayView( T * const restrict inputData,
                                   INDEX_TYPE const * const restrict inputDimensions,
                                   INDEX_TYPE const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputDimensions),
    m_strides(inputStrides)
  {}


  inline constexpr ArrayView( ManagedArray<T,NDIM,INDEX_TYPE> const & ManagedArray ):
    m_data(ManagedArray.m_data),
    m_dims(ManagedArray.m_dims),
    m_strides(ManagedArray.m_strides)
  {}

  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and calculates stride.
   */
//  inline explicit constexpr ArrayView( T * const restrict inputData ):
//    m_data(    inputData + maxDim() ),
//    m_dims( reinterpret_cast<INDEX_TYPE*>( inputData ) + 1 )
//    m_strides()
//  {}
//
  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArrayView<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the beginning of the data associated with its
   * sub-array.
   */
#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1> const operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U!=2 , ArrayView<T,NDIM-1> const >::type  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T const * restrict >::type operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }
#endif



#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1> operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U!=2 , ArrayView<T,NDIM-1> >::type  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T,NDIM-1>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T * restrict >::type operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }
#endif

  template< typename...INDICES >
  inline constexpr T & operator()( INDICES... indices ) const
  {
    return m_data[ index(indices...) ];
  }

  template< typename... INDICES >
  inline constexpr INDEX_TYPE index( INDICES... indices ) const
  {
    return index_helper<NDIM,INDICES...>::f(m_strides,indices...);
  }


private:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const * const restrict m_dims;

  INDEX_TYPE const * const restrict m_strides;


  template< int DIM, typename INDEX, typename...REMAINING_INDICES >
  struct index_helper
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                         INDEX index,
                                         REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename...REMAINING_INDICES >
  struct index_helper<1,INDEX,REMAINING_INDICES...>
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
                                         INDEX index,
                                         REMAINING_INDICES... indices )
    {
      return index;
    }
  };

};









template< typename T, typename INDEX_TYPE >
class ArrayView<T,2,INDEX_TYPE>
{
public:
  inline explicit constexpr ArrayView( T * const restrict inputData,
                                       INDEX_TYPE const * const restrict inputDimensions,
                                       INDEX_TYPE const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputDimensions),
    m_strides(inputStrides)
  {}


  inline constexpr ArrayView( ManagedArray<T,2,INDEX_TYPE> const & ManagedArray ):
    m_data(ManagedArray.m_data),
    m_dims(ManagedArray.m_dims),
    m_strides(ManagedArray.m_strides)
  {}


  inline constexpr T const * restrict operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ] );
  }

  inline constexpr T * restrict operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ] );
  }

  inline constexpr T & operator()( INDEX_TYPE index0, INDEX_TYPE index1 ) const
  {
    return m_data[index0*m_strides[0]+index1];
  }

private:
  T * const restrict m_data;
  INDEX_TYPE const * const restrict m_dims;
  INDEX_TYPE const * const restrict m_strides;
};


/**
 * Specialization for the ArrayView<typename T, int NDIM> class template. This specialization defines the lowest
 * level ArrayView in the array hierarchy. Thus this is the only level that actually allows data access, where the
 * other template instantiations only return references to a sub-array. In essence, this is a 1D array.
 */
template< typename T, typename INDEX_TYPE >
class ArrayView<T,1,INDEX_TYPE>
{
public:


  /**
   * @param data pointer to the beginning of the data
   * @param length pointer to the beginning of an array of lengths. This array has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers. Unlike the higher dimensionality arrays,
   * no calculation of stride is necessary for NDIM=1.
   */
  ArrayView( T * const restrict inputData,
         INDEX_TYPE const * const restrict inputDimensions,
         INDEX_TYPE const * const restrict ):
    m_data(inputData),
    m_dims(inputDimensions)
  {}

  ArrayView() = delete;
//  ArrayView( ArrayView const & ) = default;
//  ArrayView( ArrayView && ) = default;
//  ArrayView & operator=( ArrayView const & ) = default;
//  ArrayView & operator=( ArrayView && ) = default;
//  ~ArrayView() = default;


  /**
   * @param index index of the element in array to access
   * @return a reference to the m_data[index], where m_data is a T*.
   * This function simply returns a reference to the pointer deferenced using index.
   */
  inline constexpr T const & operator[]( INDEX_TYPE const index) const
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_dims[0] );
#endif
    return m_data[index];
  }

  inline constexpr T & operator[]( INDEX_TYPE const index)
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_dims[0] );
#endif
    return m_data[index];
  }

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const * const restrict m_dims;
};








































} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
