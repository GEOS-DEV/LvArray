/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef MANAGED_ARRAY_HPP_
#define MANAGED_ARRAY_HPP_
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

#include "ArrayView.hpp"


namespace multidimensionalArray
{



//template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
//class ArrayView;

#if 1


template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
class ManagedArray
{
public:

  using value_type = T;
  using index_type = INDEX_TYPE;
//  using unsigned_index_type = std::make_unsigned<INDEX_TYPE>;

  using Index_Sequence = std::make_index_sequence<NDIM>;
  using iterator = T*;
  using const_iterator = T const *;
  using pointer = T*;
  using const_pointer = T const *;
  using reference = T&;
  using const_reference = T const &;

  using size_type = INDEX_TYPE;



  inline explicit constexpr ManagedArray():
    dataVector(),
    m_data(nullptr),
    m_dims{0},
    m_strides{0},
    m_singleParameterResizeIndex(0)
  {}

  template< typename... DIMS >
  inline explicit constexpr ManagedArray( DIMS... dims ):
    dataVector(),
    m_data(),
    m_dims{ static_cast<INDEX_TYPE>(dims)...},
    m_strides(),
    m_singleParameterResizeIndex(0)
  {
    static_assert( sizeof...(DIMS) == NDIM , "Error: calling ManagedArray::ManagedArray with incorrect number of arguments.");
//    static_assert( check_dim_type<0,DIMS...>::value, "arguments to constructor of geosx::ManagedArray( DIMS... dims ) differ from INDEX_TYPE" );

    CalculateStrides();

    resize( dims...);

  }

  ManagedArray( ManagedArray const & source ) = default;
  ManagedArray( ManagedArray && source ) = default;

  ManagedArray & operator=( ManagedArray const & source )
  {

  }

  template< typename T_RHS >
  ManagedArray & operator=( T_RHS const & rhs )
  {
    INDEX_TYPE const length = size();
    for( INDEX_TYPE a=0 ; a<length ; ++a )
    {
      m_data[a] = rhs;
    }
    return *this;
  }


  void CalculateStrides()
  {
    m_strides[NDIM-1] = 1;
    for( int a=NDIM-2 ; a>=0 ; --a )
    {
      m_strides[a] = m_dims[a+1] * m_strides[a+1];
    }
  }


  /**
   * \defgroup stl container interface
   * @{
   */


  template< typename... DIMS >
  void resize( DIMS... newdims )
  {
    static_assert( sizeof...(DIMS) == NDIM , "Error: calling template< typename... DIMS > ManagedArray::resize(DIMS...newdims) with incorrect number of arguments.");
//    static_assert( check_dim_type<0,DIMS...>::value, "arguments to geosx::resize( DIMS... dims ) differ from INDEX_TYPE" );
    INDEX_TYPE length = 1;

    dim_unpack<0,DIMS...>::f( m_dims, newdims...);

    resize();
  }

  void resize()
  {
    INDEX_TYPE length = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      length *= m_dims[a];
    }

    dataVector.resize( length );
    m_data = dataVector.data();
  }

  template< typename TYPE >
  void resize( TYPE newdim )
  {
    m_dims[m_singleParameterResizeIndex] = newdim;
    resize();
  }




  void reserve( INDEX_TYPE newLength )
  {
    dataVector.reserve(newLength);
  }


  INDEX_TYPE size() const
  {
    return size_helper<0>::f(m_dims);
  }

  bool empty() const
  {
    return size()==0 ? true : false;
  }

  void clear()
  {
    dataVector.clear();
    m_data = nullptr;
  }

  void erase( iterator index )
  {
    dataVector.erase(static_cast<typename std::vector<T>::iterator>(index));
    m_data = dataVector.data();
  }


  reference       front()       { return dataVector.front(); }
  const_reference front() const { return dataVector.front(); }

  reference       back()       { return dataVector.back(); }
  const_reference back() const { return dataVector.back(); }

  T *       data()       {return m_data;}
  T const * data() const {return m_data;}

  iterator begin() {return m_data;}
  const_iterator begin() const {return m_data;}

  iterator end() {return &(m_data[size()]);}
  const_iterator end() const {return &(m_data[size()]);}




  void push_back( T const & newValue )
  {
    dataVector.push_back(newValue);
    m_data = dataVector.data();
  }

  void pop_back()
  {
    dataVector.pop_back();
  }

  template< class InputIt >
  void insert( iterator pos, InputIt first, InputIt last)
  {
    dataVector.insert(static_cast<typename std::vector<T>::iterator>(pos),first,last);
  }
  /**@}*/


//
//  iterator end() {return &(dataVector[size()]);}
//  const_iterator end() const {return &(dataVector[size()]);}

  INDEX_TYPE Dimension( ptrdiff_t dim ) const
  {
    return m_dims[dim];
  }

  inline constexpr ArrayView<T,NDIM,INDEX_TYPE> View() const
  {
//    return ArrayView<T,NDIM>(*this);
    return ArrayView<T,NDIM,INDEX_TYPE>(this->m_data,
                                        this->m_dims,
                                        this->m_strides);
  }

  inline constexpr ArrayView<T,NDIM,INDEX_TYPE> View()
  {
    return ArrayView<T,NDIM,INDEX_TYPE>(this->m_data,
                                        this->m_dims,
                                        this->m_strides);
  }


//  inline explicit constexpr ArrayView( T * const restrict inputData ):
//    m_data(    inputData + maxDim() ),
//    m_dims( reinterpret_cast<INDEX_TYPE*>( inputData ) + 1 )
//    m_strides()
//  {}

  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArrayView<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the beginning of the data associated with its
   * sub-array.
   */

#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1,INDEX_TYPE> const operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U >= 3 , ArrayView<T,NDIM-1,INDEX_TYPE> const >::type  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T const * restrict >::type operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1 , T const & >::type operator[](INDEX_TYPE const index) const
  {
    return m_data[ index*m_strides[0] ];
  }

#endif



#if ARRAY_BOUNDS_CHECK == 1
  inline constexpr ArrayView<T,NDIM-1,INDEX_TYPE> operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U>=3 , ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2 , T * restrict >::type
  operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1 , T & >::type
  operator[](INDEX_TYPE const index)
  {
    return m_data[ index*m_strides[0] ];
  }

#endif





  template< typename...DIMS >
  inline constexpr T & operator()( DIMS... dims ) const
  {
    return m_data[ linearIndex(dims...) ];
  }

  template< typename... DIMS >
  inline constexpr INDEX_TYPE linearIndex( DIMS... dims ) const
  {
    return index_helper<NDIM,DIMS...>::f(m_strides,dims...);
  }


//private:
  std::vector<T> dataVector;

  /// pointer to beginning of data for this array, or sub-array.
  T * restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE m_dims[NDIM];

  INDEX_TYPE m_strides[NDIM];

  int m_singleParameterResizeIndex = 0;



  template< int INDEX, typename DIM0, typename... DIMS >
  struct check_dim_type
  {
    constexpr static bool value =  std::is_same<DIM0,INDEX_TYPE>::value && check_dim_type<INDEX+1, DIMS...>::value;
  };

  template< typename DIM0, typename... DIMS >
  struct check_dim_type<NDIM-1,DIM0,DIMS...>
  {
    constexpr static bool value = std::is_same<DIM0,INDEX_TYPE>::value;
  };



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




  template< int INDEX, typename DIM0, typename... DIMS >
  struct dim_unpack
  {
    constexpr static void f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[INDEX] = dim0;
      dim_unpack< INDEX+1, DIMS...>::f( m_dims, dims... );
    }
  };

  template< typename DIM0, typename... DIMS >
  struct dim_unpack<NDIM-1,DIM0,DIMS...>
  {
    constexpr static void f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[NDIM-1] = dim0;
      return ;
    }

  };


  template< int DIM >
  struct size_helper
  {
    template< int INDEX=DIM >
    constexpr static typename std::enable_if<INDEX!=NDIM-1,INDEX_TYPE>::type f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX] * size_helper<INDEX+1>::f(dims);
    }
    template< int INDEX=DIM >
    constexpr static typename std::enable_if<INDEX==NDIM-1,INDEX_TYPE>::type f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX] ;
    }

  };
};
#else

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

template< typename T, int NDIM, typename INDEX_TYPE, typename INDEX_SEQUENCE >
class ManagedArray_impl {};

template < typename T, int NDIM , typename INDEX_TYPE = std::int_fast32_t >
using ManagedArray = ManagedArray_impl<T, NDIM, INDEX_TYPE, std::make_index_sequence<NDIM> >;


template< typename T, int NDIM, typename INDEX_TYPE, size_t... INDEX_SEQUENCE >
class ManagedArray_impl<T, NDIM, INDEX_TYPE, std::index_sequence<INDEX_SEQUENCE...> >
{
public:

  using Index_Sequence = std::make_index_sequence<NDIM>;

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
#endif

















} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
