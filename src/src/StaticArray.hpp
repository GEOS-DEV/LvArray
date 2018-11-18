/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
 * @file StaticArray.hpp
 */

#ifndef STATICARRAY_HPP_
#define STATICARRAY_HPP_

#include "ArraySlice.hpp"
#include "ArrayView.hpp" // for integer_conversion<>() TODO remove
#include "Array.hpp"     // for is_integer<>()         TODO remove

#include <array>

namespace LvArray
{

/**
 * @class StaticArray
 * @brief This class represents a static-size array with local (stack) memory
 * @tparam T type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 * @tparam MAXSIZE a list of maximum array size for each dimension
 *
 * The main objective of this class is to wrap around and provide an ArraySlice interface to a stack-based T[] array.
 * This can be quite handy in various kernels, where local objects with multi-dim array functionality are needed,
 * but using Array class is not possible because of its dynamic allocation. Here the amount of memory to be allocated
 * must be known at compile-time; "resizing" this array does not change that amount, and the array cannot be resized
 * beyond this static capacity.
 */
template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t, INDEX_TYPE ... MAXSIZE >
class StaticArray : public ArraySlice<T, NDIM, INDEX_TYPE >
{
protected:

  static_assert( sizeof...(MAXSIZE) == NDIM, "StaticArray: number of max sizes must be equal to number of dimensions" );

  template< typename INDEX, INDEX size, INDEX ... sizes >
  static constexpr INDEX static_size_helper()
  {
    return size * static_size_helper<INDEX, sizes...>();
  }

  template< typename INDEX >
  static constexpr INDEX static_size_helper()
  {
    return 1;
  }

public:

  using ArraySlice<T, NDIM, INDEX_TYPE>::m_data;
  using ArraySlice<T, NDIM, INDEX_TYPE>::m_dims;
  using ArraySlice<T, NDIM, INDEX_TYPE>::m_strides;

  static constexpr INDEX_TYPE max_size = static_size_helper<INDEX_TYPE, MAXSIZE...>();

  using ArrayType = std::array<T, max_size>;
  using pointer = T *;
  using const_pointer = T const *;
  using iterator = typename ArrayType::iterator;
  using const_iterator = typename ArrayType::const_iterator;

  /**
   * The default constructor
   */
  inline explicit CONSTEXPRFUNC
  StaticArray():
    ArraySlice<T, NDIM, INDEX_TYPE>( m_dataMem.data(), m_dimsMem, m_stridesMem )
  {

  }

  /**
   * @brief constructor that takes in the dimensions as a variadic parameter list
   * @param dims the dimensions of the array in form ( n0, n1,..., n(NDIM-1) )
   */
  template< typename... DIMS >
  inline explicit StaticArray( DIMS... dims ):
    StaticArray()
  {
    static_assert( is_integer<INDEX_TYPE>::value, "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling StaticArray::StaticArray with incorrect number of arguments." );
    static_assert( check_dim_type<0, DIMS...>::value, "arguments to constructor of StaticArray::StaticArray( DIMS... dims ) are incompatible with INDEX_TYPE" );

    resize( dims... );
  }

  /**
   * @brief copy constructor
   * @param source object to copy
   *
   * Performs a deep copy of source
   */
  StaticArray( StaticArray const & source ):
    StaticArray()
  {
    *this = source;
  }

  /**
   * @brief move constructor is deleted
   */
  StaticArray( StaticArray && source ) = delete;

  /**
   * Destructor - nothing to do
   */
  ~StaticArray() = default;

  /**
   * @brief assignment operator
   * @param rhs source for the assignment
   * @return *this
   *
   * The assignment operator performs a deep copy of the rhs.
   */
  StaticArray & operator=( StaticArray const& rhs )
  {
    resize( NDIM, rhs.m_dimsMem );

    INDEX_TYPE const length = size();
    T * const data_ptr = data();
    T const * const rhs_data_ptr = rhs.data();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      data_ptr[a] = rhs_data_ptr[a];
    }

    return *this;
  }

  /**
   * @brief move assignment is deleted
   */
  StaticArray & operator=( StaticArray && ) = delete;

  /**
   * @brief set all values of array to rhs
   * @param rhs value that array will be set to.
   */
  inline CONSTEXPRFUNC
  StaticArray & operator=( T const & rhs )
  {
    INDEX_TYPE const length = size();
    T* const data_ptr = data();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      data_ptr[a] = rhs;
    }
    return *this;
  }

  /**
   * @brief assignment operator from any ArrayView of compatible type
   * @param rhs source for the assignment
   * @return *this
   *
   * The assignment operator performs a deep copy of the rhs.
   */
  StaticArray & operator=( ArrayView<T, NDIM, INDEX_TYPE> const& rhs )
  {
    resize( NDIM, rhs.dims() );

    INDEX_TYPE const length = size();
    T * const data_ptr = data();
    T const * const rhs_data_ptr = rhs.data();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      data_ptr[a] = rhs_data_ptr[a];
    }

    return *this;
  }

  int numDimensions() const
  { return NDIM; }

  /**
   * @brief This function provides a resize or reallocation of the array
   * @param numDims the number of dims in the dims parameter
   * @param dims the new size of the dimensions
   */
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE const * const dims )
  {
    GEOS_ERROR_IF( numDims != NDIM, "Dimension mismatch: " << numDims );
    this->setDims( dims );
    CalculateStrides();
    resize();
  }

  /**
   * @brief This function provides a resize or reallocation of the array
   * @param numDims the number of dims in the dims parameter
   * @param dims the new size of the dimensions
   *
   * @note this is required due to an issue where some compilers may prefer the full variadic
   *       parameter pack template<typename ... DIMS> resize( DIMS... newdims)
   */
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE * const dims )
  {
    resize( numDims, const_cast<DIMS_TYPE const * const>(dims) );
  }

  /**
   * @brief function to resize/reallocate the array
   * @tparam DIMS variadic pack containing the dimension types
   * @param newdims the new dimensions
   */
  template< typename... DIMS >
  void resize( DIMS... newdims )
  {
    static_assert( sizeof ... (DIMS) == NDIM,
                   "Error: calling template< typename... DIMS > Array::resize(DIMS...newdims) with incorrect number of arguments." );
    static_assert( check_dim_type<0, DIMS...>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    dim_unpack<0, DIMS...>::f( m_dimsMem, newdims... );
    CalculateStrides();
    resize();
  }

  /**
 * User Defined Conversion operator to move from an StaticArray<T> to StaticArray<T const>&.  This is
 * achieved by applying a reinterpret_cast to the this pointer, which is a safe operation as the
 * only difference between the types is a const specifier.
 */
  template< typename U = T >
  inline CONSTEXPRFUNC
  operator typename std::enable_if< !std::is_const<U>::value,
    StaticArray<T const, NDIM, INDEX_TYPE, MAXSIZE...> const & >::type
  () const
  {
    return reinterpret_cast<StaticArray<T const, NDIM, INDEX_TYPE, MAXSIZE...> const &>(*this);
  }

  /**
   * @brief function to return the allocated size
   */
  inline LVARRAY_HOST_DEVICE INDEX_TYPE size() const
  {
    return size_helper<0>::f( m_dimsMem );
  }

  /**
   * @brief function check if the array is empty.
   * @return a boolean. True if the array is empty, False if it is not empty.
   */
  inline bool empty() const
  {
    return size() == 0;
  }

  /**
   * @brief std::vector-like front method
   * @return a reference to the first element of the array
   */
  template< typename U = T >
  inline T& front()
  {
    return m_dataMem.front();
  }

  /**
   * @brief std::vector-like front method
   * @return a reference to the first element of the array
   */
  inline T const& front() const
  {
    return m_dataMem.front();
  }

  /**
   * @brief std::vector-like back method
   * @return a reference to the last element of the array
   */
  inline T& back()
  {
    return m_dataMem.back();
  }

  /**
   * @brief std::vector-like back method
   * @return a reference to the last element of the array
   */
  inline T const& back() const
  {
    return m_dataMem.back();
  }

  /**
   * @brief std::vector-like begin method
   * @return an iterator to the first element of the array
   */
  inline iterator begin()
  {
    return m_dataMem.begin();
  }

  /**
   * @brief std::vector-like begin method
   * @return an iterator to the first element of the array
   */
  inline const_iterator begin() const
  {
    return m_dataMem.begin();
  }

  /**
   * @brief std::vector-like end method
   * @return an iterator to the element of the array past the last element
   */
  inline iterator end()
  {
    return m_dataMem.end();
  }

  /**
   * @brief std::vector-like end method
   * @return an iterator to the element of the array past the last element
   */
  inline const_iterator end() const
  {
    return m_dataMem.end();
  }

  /**
   * @brief operator() array accessor
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request (0,3,4)
   * @return reference to the data at the requested indices.
   *
   * This is a standard fortran like parentheses interface to array access.
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_data[ linearIndex( indices... ) ];
  }

  /**
   * @brief calculation of offset or linear index from a multidimensional space to a linear space.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request (0,3,4)
   *
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    index_checker<NDIM, INDICES...>::f( m_dimsMem, indices... );
#endif
    return index_helper<NDIM, INDICES...>::f( m_stridesMem, indices... );
  }

  /**
   * @brief accessor for data
   * @return pointer to the data
   */
  inline T * data()
  {
    return m_dataMem.data();
  }

  /**
   * @brief accessor for data
   * @return pointer to const to the data
   */
  inline T const * data() const
  {
    return m_dataMem.data();
  }

  /**
   * @brief function to get a pointer to a slice of data
   * @param index the index of the slice to get
   * @return
   * @todo THIS FUNCION NEEDS TO BE GENERALIZED for all dims
   */
  inline T const * data( INDEX_TYPE const index ) const
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return &(m_dataMem[ index*m_stridesMem[0] ]);
  }

  /**
   * @brief function to get a pointer to a slice of data
   * @param index the index of the slice to get
   * @return
   * @todo THIS FUNCION NEEDS TO BE GENERALIZED for all dims
   */
  inline T * data( INDEX_TYPE const index )
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return &(m_dataMem[ index*m_stridesMem[0] ]);
  }

  /**
   * @brief function to get the length of a dimension
   * @param dim the dimension for which to get the length of
   */
  inline LVARRAY_HOST_DEVICE INDEX_TYPE size( int dim ) const
  {
    return m_dimsMem[dim];
  }

  /**
   * @brief this function is an accessor for the dims array
   * @return a pointer to m_dimsMem
   */
  inline INDEX_TYPE const * dims() const
  {
    return m_dimsMem;
  }

  /**
   * @brief this function is an accessor for the strides array
   * @return a pointer to m_stridesMem
   */
  inline INDEX_TYPE const * strides() const
  {
    return m_stridesMem;
  }

  /**
   * @brief This function outputs the contents of an array to an output stream
   * @param stream the output stream for which to apply operator<<
   * @param array the array to output
   * @return a reference to the ostream
   */
  friend std::ostream& operator<< ( std::ostream& stream, StaticArray const & array )
  {
    T const * const data_ptr = array.data();
    stream<<"{ "<< data_ptr[0];
    for( INDEX_TYPE a=1 ; a<array.size() ; ++a )
    {
      stream<<", "<< data_ptr[a];
    }
    stream<<" }";
    return stream;
  }

protected:

  void resize()
  {
    GEOS_ERROR_IF( size() > max_size, "StaticArray: requested size exceeds available static memory" );
  }

  void CalculateStrides()
  {
    m_stridesMem[NDIM-1] = 1;
    for( int a=NDIM-2 ; a>=0 ; --a )
    {
      m_stridesMem[a] = m_dimsMem[a+1] * m_stridesMem[a+1];
    }
  }

  /**
   * @brief this function sets the dimensions of the array, but does not perform a resize to the
   *        new dimensions
   * @param dims a pointer/array containing the the new dimensions
   */
  template< typename DIMS_TYPE >
  void setDims( DIMS_TYPE const dims[NDIM] )
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      this->m_dimsMem[a] = integer_conversion<INDEX_TYPE>(dims[a]);
    }
  }

  /**
   * @brief this function sets the strides of the array.
   * @param strides a pointer/array containing the the new strides
   */
  void setStrides( INDEX_TYPE const strides[NDIM] )
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      this->m_stridesMem[a] = strides[a];
    }
  }

  template< typename CANDIDATE_INDEX_TYPE >
  struct is_valid_indexType
  {
    constexpr static bool value = std::is_same<CANDIDATE_INDEX_TYPE, INDEX_TYPE>::value ||
                                  ( is_integer<CANDIDATE_INDEX_TYPE>::value &&
                                    ( sizeof(CANDIDATE_INDEX_TYPE)<=sizeof(INDEX_TYPE) ) );
  };
  template< int INDEX, typename DIM0, typename... DIMS >
  struct check_dim_type
  {
    constexpr static bool value =  is_valid_indexType<DIM0>::value && check_dim_type<INDEX+1, DIMS...>::value;
  };

  template< typename DIM0, typename... DIMS >
  struct check_dim_type<NDIM-1, DIM0, DIMS...>
  {
    constexpr static bool value = is_valid_indexType<DIM0>::value;
  };

  template< int INDEX, typename DIM0, typename... DIMS >
  struct dim_unpack
  {
    constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[INDEX] = dim0;
      dim_unpack< INDEX+1, DIMS...>::f( m_dims, dims... );
      return 0;
    }
  };

  template< typename DIM0, typename... DIMS >
  struct dim_unpack<NDIM-1, DIM0, DIMS...>
  {
    constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... )
    {
      m_dims[NDIM-1] = dim0;
      return 0;
    }
  };

  /**
   * @struct This is a functor to calculate the linear index of a multidimensional array.
   */
  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_helper
  {
    LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static INDEX_TYPE
    f( INDEX_TYPE const * const restrict strides,
       INDEX index, REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1, REMAINING_INDICES...>::f( strides+1, indices... );
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_helper<1, INDEX, REMAINING_INDICES...>
  {
    LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static INDEX_TYPE
    f( INDEX_TYPE const * const restrict,
       INDEX index )
    {
      return index;
    }
  };

#ifdef USE_ARRAY_BOUNDS_CHECK
  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_checker
  {
    LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static void
    f( INDEX_TYPE const * const restrict dims,
       INDEX index, REMAINING_INDICES... indices )
    {
      GEOS_ERROR_IF( index < 0 || index > dims[0], "index=" << index << ", m_dims[" <<
                                                            (NDIM - DIM) << "]=" << dims[0] );
      index_checker<DIM-1, REMAINING_INDICES...>::f( dims + 1, indices... );
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_checker<1, INDEX, REMAINING_INDICES...>
  {
    LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC static void
    f( INDEX_TYPE const * const restrict dims,
       INDEX index )
    {
      GEOS_ERROR_IF( index < 0 || index > dims[0], "index=" << index << ", m_dims[" <<
                                                            (NDIM - 1) << "]=" << dims[0] );
    }
  };
#endif

  /**
   * @struct this is a functor to calculate the total size of the array from the dimensions.
   */
  template< int DIM >
  struct size_helper
  {
    template< int INDEX=DIM >
    inline CONSTEXPRFUNC static typename std::enable_if<INDEX!=NDIM-1, INDEX_TYPE>::type
    f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX] * size_helper<INDEX+1>::f( dims );
    }

    template< int INDEX=DIM >
    inline CONSTEXPRFUNC static typename std::enable_if<INDEX==NDIM-1, INDEX_TYPE>::type
    f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX];
    }

  };


  /// this data member contains the dimensions of the array
  INDEX_TYPE m_dimsMem[NDIM];

  /// this data member contains the strides of the array
  INDEX_TYPE m_stridesMem[NDIM];

  /// this data member contains the actual data stored by the array
  ArrayType m_dataMem;

};

}

#endif //STATICARRAY_HPP_
