/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef MANAGED_ARRAY_HPP_
#define MANAGED_ARRAY_HPP_

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif

#include <iostream>
#include <limits>
#include <vector>
//#include <utility>

#include "ArrayView.hpp"

template< typename T >
struct is_integer
{
  constexpr static bool value = std::is_same<T,int>::value ||
                                std::is_same<T,unsigned int>::value ||
                                std::is_same<T,long int>::value ||
                                std::is_same<T,unsigned long int>::value ||
                                std::is_same<T,long long int>::value ||
                                std::is_same<T,unsigned long long int>::value;
};


namespace multidimensionalArray
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"


template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_unsigned<T>::value && std::is_signed<RTYPE>::value, RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max()  )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}

template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_signed<T>::value && std::is_unsigned<RTYPE>::value, RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max() ||
      input < 0 )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}


template< typename RTYPE, typename T >
inline typename std::enable_if< ( std::is_signed<T>::value && std::is_signed<RTYPE>::value ) ||
                         ( std::is_unsigned<T>::value && std::is_unsigned<RTYPE>::value ), RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max() ||
      input < std::numeric_limits<RTYPE>::lowest() )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}



#pragma GCC diagnostic pop


//template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
//class ArrayView;


template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
class ManagedArray
{
public:

  using value_type = T;
  using index_type = INDEX_TYPE;
//  using unsigned_index_type = std::make_unsigned<INDEX_TYPE>;

  using Index_Sequence = std::make_index_sequence<NDIM>;

//  using iterator = T*;
//  using const_iterator = T const *;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;


  using pointer = T*;
  using const_pointer = T const *;
  using reference = T&;
  using const_reference = T const &;

  using size_type = INDEX_TYPE;


  inline ManagedArray():
    dataVector(),
    m_data(nullptr),
    m_dims{0},
    m_strides{0},
    m_singleParameterResizeIndex(0)
  {}

  template< typename... DIMS >
  inline explicit ManagedArray( DIMS... dims ):
    dataVector(),
    m_data(),
    m_dims{ static_cast<INDEX_TYPE>(dims) ...},
    m_strides(),
    m_singleParameterResizeIndex(0)
  {
    static_assert( is_integer<INDEX_TYPE>::value, "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling ManagedArray::ManagedArray with incorrect number of arguments.");
    static_assert( check_dim_type<0,DIMS...>::value, "arguments to constructor of geosx::ManagedArray( DIMS... dims ) are incompatible with INDEX_TYPE" );
    CalculateStrides();

    resize( dims...);

  }


  ManagedArray( ManagedArray const & source ):
    dataVector(source.dataVector),
    m_data(dataVector.data()),
    m_dims(),
    m_strides(),
    m_singleParameterResizeIndex(source.m_singleParameterResizeIndex)
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a]     = source.m_dims[a];
      m_strides[a]  = source.m_strides[a];
    }
  }


//  ManagedArray( ManagedArray && ) = delete;
//  ManagedArray & operator=( ManagedArray && ) =default;
//  ManagedArray( ManagedArray && source ):
//    dataVector(std::move(source.dataVector)),
//    m_data(dataVector.data()),
//    m_dims(source.m_dims),
//    m_strides(source.m_strides)
//  {
//    for( int a=0 ; a<NDIM ; ++a )
//    {
//      m_dims[a]     = m_dims[a];
//      m_strides[a]  = m_strides[a];
//    }
//  }


  ManagedArray & operator=( ManagedArray const & source )
  {
    dataVector = source.dataVector;
    m_data     = dataVector.data();
    m_singleParameterResizeIndex = source.m_singleParameterResizeIndex;

    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a]     = source.m_dims[a];
      m_strides[a]  = source.m_strides[a];
    }

    return *this;
  }

  ManagedArray & operator=( T const & rhs )
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
    static_assert( sizeof ... (DIMS) == NDIM,
                   "Error: calling template< typename... DIMS > ManagedArray::resize(DIMS...newdims) with incorrect number of arguments.");
    static_assert( check_dim_type<0,DIMS...>::value, "arguments to ManagedArray::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE length = 1;

    dim_unpack<0,DIMS...>::f( m_dims, newdims...);
    CalculateStrides();
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
    static_assert( is_valid_indexType<TYPE>::value, "arguments to ManagedArray::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    m_dims[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resize();
  }



  void reserve( INDEX_TYPE newLength )
  {
    dataVector.reserve(newLength);
  }



#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

  /**
   *
   * @return total length of the array across all dimensions
   * @note if this function is called from a function/method where the
   * ManagedArray is a template parameter, and the template type is specified
   * then the keyword "template" must be inserted prior to size<>(). For
   * instance:
   *
   * template<typename T>
   * static int f( ManagedArray<T>& a )
   * {
   *  return a.size<int>();          // error: use 'template' keyword to treat
   *                                 // 'size' as a dependent template name
   *  return a.template size<int>(); // OK
   *
   *  return a.size();               //OK
   * }
   *
   */
  template< typename RTYPE = INDEX_TYPE >
  RTYPE size() const
  {
    if( size_helper<0>::f(m_dims) > std::numeric_limits<RTYPE>::max() ||
        size_helper<0>::f(m_dims) < std::numeric_limits<RTYPE>::lowest() )
    {
      abort();
    }
    return static_cast<RTYPE>(size_helper<0>::f(m_dims));
  }


  /**
   *
   * @param dim dimension for which the size is requested
   * @return length of a single dimensions specified by dim
   * @note if this function is called from a function/method where the
   * ManagedArray is a template parameter, and the template type is specified
   * then the keyword "template" must be inserted prior to size<>(). For
   * instance:
   *
   * template<typename T>
   * static int f( ManagedArray<T>& a )
   * {
   *  return a.size<int>(1);          // error: use 'template' keyword to treat
   *                                 // 'size' as a dependent template name
   *  return a.template size<int>(1); // OK
   *
   *  return a.size(1);               //OK
   * }
   *
   */
  template< typename RTYPE = INDEX_TYPE >
  RTYPE size( int dim ) const
  {
    if( m_dims[dim] > std::numeric_limits<RTYPE>::max() ||
        m_dims[dim] < std::numeric_limits<RTYPE>::lowest() )
    {
      abort();
    }
    return static_cast<RTYPE>(m_dims[dim]);
  }
#pragma GCC diagnostic pop


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
    dataVector.erase( index ) ;
    m_data = dataVector.data();
  }


  reference       front()       { return dataVector.front(); }
  const_reference front() const { return dataVector.front(); }

  reference       back()       { return dataVector.back(); }
  const_reference back() const { return dataVector.back(); }

  T *       data()       {return m_data;}
  T const * data() const {return m_data;}

//  iterator begin() {return m_data;}
//  const_iterator begin() const {return m_data;}
//
//  iterator end() {return &(m_data[size()]);}
//  const_iterator end() const {return &(m_data[size()]);}
  iterator begin() {return dataVector.begin();}
  const_iterator begin() const {return dataVector.begin();}

  iterator end() {return dataVector.end();}
  const_iterator end() const {return dataVector.end();}



  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type push_back( T const & newValue )
  {
    dataVector.push_back(newValue);
    m_data = dataVector.data();
    m_dims[0] = integer_conversion<INDEX_TYPE>(dataVector.size());
    CalculateStrides();
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type pop_back()
  {
    dataVector.pop_back();
    m_dims[0] = integer_conversion<INDEX_TYPE>(dataVector.size());
    CalculateStrides();
  }

  template< int N=NDIM, class InputIt >
  typename std::enable_if< N==1, void >::type insert( const_iterator pos, InputIt first, InputIt last)
  {
//    dataVector.insert(static_cast<typename std::vector<T>::iterator>(pos),first,last);
//    std::vector<T> junk;
//    dataVector.insert( dataVector.end(), junk.begin(), junk.end() );
    dataVector.insert( pos, first, last );
    m_dims[0] = integer_conversion<INDEX_TYPE>(dataVector.size());
    CalculateStrides();
  }
  /**@}*/


//
//  iterator end() {return &(dataVector[size()]);}
//  const_iterator end() const {return &(dataVector[size()]);}


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
  inline constexpr typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return m_data[ index ];
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U >= 3, ArrayView<T,NDIM-1,INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2, T const * restrict >::type
  operator[](INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    return m_data[ index ];
  }

#endif



#if ARRAY_BOUNDS_CHECK == 1
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U!=1, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1, T & >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return m_data[ index ];
  }
#else
  template< int U=NDIM >
  inline constexpr typename std::enable_if< U>=3, ArrayView<T,NDIM-1,INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==2, T * restrict >::type
  operator[](INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  template< int U=NDIM >
  inline constexpr typename std::enable_if< U==1, T & >::type
  operator[](INDEX_TYPE const index)
  {
    return m_data[ index ];
  }

#endif



  template< typename... DIMS >
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

  /// pointer to array of length NDIM that contains the lengths of each array
  // dimension
  INDEX_TYPE m_dims[NDIM];

  INDEX_TYPE m_strides[NDIM];

  int m_singleParameterResizeIndex = 0;


  template< typename CANDIDATE_INDEX_TYPE >
  struct is_valid_indexType
  {
    constexpr static bool value = std::is_same<CANDIDATE_INDEX_TYPE,INDEX_TYPE>::value ||
                                  ( is_integer<CANDIDATE_INDEX_TYPE>::value &&
                                    ( sizeof(CANDIDATE_INDEX_TYPE)<=sizeof(INDEX_TYPE) ) );
  };


  template< int INDEX, typename DIM0, typename... DIMS >
  struct check_dim_type
  {
    constexpr static bool value =  is_valid_indexType<DIM0>::value && check_dim_type<INDEX+1, DIMS...>::value;
  };

  template< typename DIM0, typename... DIMS >
  struct check_dim_type<NDIM-1,DIM0,DIMS...>
  {
    constexpr static bool value = is_valid_indexType<DIM0>::value;
  };



  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_helper
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                          INDEX index,
                                          REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
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
      return dims[INDEX];
    }

  };
};



} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
