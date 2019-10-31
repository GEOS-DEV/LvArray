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
 * @file ArrayView.hpp
 */

#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_


#include "ArraySlice.hpp"
#include "Logger.hpp"
#include "ChaiVector.hpp"
#include "helpers.hpp"
#include "IntegerConversion.hpp"
#include "SFINAE_Macros.hpp"

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
#include "totalview/tv_helpers.hpp"
#include "totalview/tv_data_display.h"
#endif

namespace LvArray
{



template< typename T,
          int NDIM,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
class Array;

/**
 * @class ArrayView
 * @brief This class serves to provide a "view" of a multidimensional array.
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 *
 *
 * When using CHAI the copy copy constructor of this class calls the copy constructor for CHAI
 * which will move the data to the location of the touch (host or device). In general, the
 * ArrayView should be what is passed by value into a lambda that is used to launch kernels as it
 * copy will trigger the desired data motion onto the appropriate memory space.
 *
 * Key features:
 * 1) ArrayView copy construction triggers a CHAI copy constructor
 * 2) Inherits operator[] array accessor from ArraySlice
 * 3) Defines operator() array accessor
 * 3) operator[] and operator() are all const and may be called in non-mutable lamdas
 * 4) Conversion operators to go from ArrayView<T> to ArrayView<T const>
 * 5) Since the Array is derived from ArrayView, it may be upcasted:
 *      Array<T,NDIM> array;
 *      ArrayView<T,NDIM> const & arrView = array;
 *
 * A good guideline is to upcast to an ArrayView when you don't need allocation capabilities that
 * are only present in Array.
 *
 */
template< typename T,
          int NDIM,
          typename INDEX_TYPE,
          typename DATA_VECTOR_TYPE >
class ArrayView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  using value_type = T;
  using iterator = T *;
  using const_iterator = T const *;
  using pointer = T *;
  using const_pointer = T const *;

  /**
   * The default constructor
   */
  inline explicit CONSTEXPRFUNC
  ArrayView() noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataVector()
  {
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
    ArrayView::TV_ttf_display_type( nullptr );
#endif
  }

  inline explicit CONSTEXPRFUNC
  ArrayView( std::nullptr_t ) noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataVector( nullptr )
  {}

  /**
   * @brief constructor to make a shallow copy of the input data
   * @param dimsMem
   * @param stridesMem
   * @param dataVector
   * @param singleParameterResizeIndex
   * @return
   */
  inline CONSTEXPRFUNC
  ArrayView( INDEX_TYPE const * const dimsMem,
             INDEX_TYPE const * const stridesMem,
             DATA_VECTOR_TYPE const & dataVector,
             INDEX_TYPE const CXX_UTILS_UNUSED_ARG( singleParameterResizeIndex ) ) noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataVector(nullptr)
  {
    m_dataVector = dataVector;
    setDims( dimsMem );
    setStrides( stridesMem );
    setDataPtr();
  }

  /**
   * @brief Copy Constructor
   * @param source the object to copy
   * @return
   *
   * The copy constructor will trigger the copy constructor for m_dataVector, which will trigger
   * a CHAI::ManagedArray copy construction, and thus may copy the memory to the memory space
   * associated with the copy. I.e. if passed into a lambda which is used to execute a cuda kernel,
   * the data will be allocated and copied to the device (if it doesn't already exist).
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArrayView( ArrayView const & source ) noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataVector{ source.m_dataVector },
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    setDataPtr();
    setDims( source.m_dims );
    setStrides( source.m_strides );
  }

  /**
   * @brief move constructor
   * @param source object to move
   * @return
   * moves source into this without triggering any copy.
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArrayView( ArrayView && source ):
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataVector( std::move( source.m_dataVector ) ),
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    setDataPtr();
    setDims( source.m_dims );
    setStrides( source.m_strides );

    source.m_dataVector.reset();
    source.setDataPtr();
  }

  /**
   * @brief copy assignment operator
   * @param rhs object to copy
   */
  inline CONSTEXPRFUNC
  ArrayView & operator=( ArrayView const & rhs ) noexcept
  {
    m_dataVector = rhs.m_dataVector;
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    setStrides( rhs.m_strides );
    setDims( rhs.m_dims );
    setDataPtr();
    return *this;
  }

  /**
   * @brief set all values of array to rhs
   * @param rhs value that array will be set to.
   */
  DISABLE_HD_WARNING
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  typename std::enable_if< !(std::is_const<U>::value), ArrayView const & >::type
  operator=( T const & rhs ) const noexcept
  {
    INDEX_TYPE const length = size();
    T* const data_ptr = data();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      data_ptr[a] = rhs;
    }
    return *this;
  }

  ArrayView & operator=( ArrayView && rhs )
  {
    *this = rhs;
    rhs.m_dataVector.reset();
    rhs.setDataPtr();
    return *this;
  }

  /**
   * User Defined Conversion operator to move from an "ArrayView<T> const" to
   * "ArrayView<T const> const &".  This is achieved by applying a reinterpret_cast to the this
   * pointer, which is a safe operation as the only difference between the types is a
   * const specifier.
   */
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator typename std::enable_if< !std::is_const<U>::value,
                                    ArrayView<T const, NDIM, INDEX_TYPE> const & >::type
    () const noexcept
  {
    return reinterpret_cast<ArrayView<T const, NDIM, INDEX_TYPE> const &>(*this);
  }

  /**
   * User Defined Conversion operator to move from an ArrayView<T> to ArraySlice<T>.  This is
   * achieved by applying a reinterpret_cast to the this pointer, which is a safe operation as the
   * only difference between the types is a const specifier.
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator ArraySlice<T, NDIM, INDEX_TYPE>() const noexcept
  {
    return ArraySlice<T, NDIM, INDEX_TYPE>( m_data, m_dims, m_strides );
  }

  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator typename std::enable_if< !std::is_const<U>::value,
                                    ArraySlice<T const, NDIM, INDEX_TYPE> const >::type
    () const noexcept
  {
    return ArraySlice<T const, NDIM, INDEX_TYPE>( m_data, m_dims, m_strides );
  }


  /**
   * User defined conversion to convert a ArraySlice<T,1,NDIM> to a raw pointer.
   * @return a copy of m_data.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline operator
  typename std::enable_if< U == 1, T * const restrict >::type () const noexcept restrict_this
  {
    return m_data;
  }

  /**
   * Function to convert to a reduced dimension array. For example, converting from
   * a 2d array to a 1d array is valid if the last dimension of the 2d array is 1.
   */
  template< int U=NDIM >
  inline CONSTEXPRFUNC
  typename std::enable_if< (U > 1), ArrayView<T, NDIM - 1, INDEX_TYPE, DATA_VECTOR_TYPE> >::type
  dimReduce() const noexcept
  {
    GEOS_ASSERT_MSG( m_dims[NDIM - 1] == 1,
                     "Array::operator ArrayView<T,NDIM-1,INDEX_TYPE> is only valid if last "
                     "dimension is equal to 1." );

    return ArrayView<T, NDIM - 1, INDEX_TYPE, DATA_VECTOR_TYPE>( this->m_dims,
                                                                 this->m_strides,
                                                                 this->m_dataVector,
                                                                 this->m_singleParameterResizeIndex );
  }


  /**
   * @brief function to return the allocated size
   */
  inline LVARRAY_HOST_DEVICE INDEX_TYPE size() const noexcept
  { return size_helper<NDIM, INDEX_TYPE>::f( m_dims ); }

  /**
   * @brief function check if the array is empty.
   * @return a boolean. True if the array is empty, False if it is not empty.
   */
  inline bool empty() const
  { return m_dataVector.empty(); }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * begin() const
  { return data(); }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * end() const
  { return data() + size(); }

  T & front() const
  { return m_data[0]; }

  T & back() const
  { return m_data[size() - 1]; }

  ///***********************************************************************************************
  ///**** DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************

  /**
   * @brief This function provides a square bracket operator for array access.
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArraySlice<T,NDIM-1>.
   *
   * This function creates a new ArraySlice<T,NDIM-1,INDEX_TYPE> by passing the member pointer
   * locations +1. Thus, the returned object has m_data pointing to the beginning of the data
   * associated with its sub-array. If used as a set of nested operators (array[][][]) the compiler
   * typically provides the pointer dereference directly and does not create the new ArraySlice
   * objects.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC typename
  std::enable_if< U >= 3, ArraySlice<T, NDIM-1, INDEX_TYPE> >::type
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice<T, NDIM-1, INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

  /**
   * @brief This function provides a square bracket operator for array access of a 2D array.
   * @param index index of the element in array to access
   * @return a pointer to the data location indicated by the index.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< U==2, ArraySlice1d_rval< T, INDEX_TYPE > >::type
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return CREATE_ARRAY_SLICE_1D( &(m_data[ index*m_strides[0] ]), m_dims+1, m_strides+1 );
  }

  /**
   * @brief This function provides a square bracket operator for array access for a 1D array.
   * @param index index of the element in array to access
   * @return a reference to the data location indicated by the index.
   *
   * @note This declaration is for NDIM==1 when bounds checking is ON.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< U==1, T & >::type
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return m_data[ index ];
  }

  ///***********************************************************************************************
  ///**** END DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************

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
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    linearIndex_helper<NDIM, INDEX_TYPE, INDICES...>::check( m_dims, indices... );
#endif
    return linearIndex_helper<NDIM, INDEX_TYPE, INDICES...>::evaluate( m_strides, indices... );
  }

  /**
   * @brief accessor for data
   * @return pointer to the data
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  { return m_dataVector.data(); }

  /**
   * @brief function to get a pointer to a slice of data
   * @param index the index of the slice to get
   * @return
   * @todo THIS FUNCION NEEDS TO BE GENERALIZED for all dims
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data( INDEX_TYPE const index ) const
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return &(m_dataVector[ index*m_strides[0] ]);
  }

  /**
   * @brief copy value from one location in this array to another
   * @param destIndex index for which to copy data into
   * @param sourceIndex index to copy data from
   */
  void copy( INDEX_TYPE const destIndex, INDEX_TYPE const sourceIndex )
  {
    ARRAY_SLICE_CHECK_BOUNDS( destIndex );
    ARRAY_SLICE_CHECK_BOUNDS( sourceIndex );

    INDEX_TYPE const stride0 = m_strides[0];
    T * const dest = data( destIndex );
    T const * const source = data( sourceIndex );

    for( INDEX_TYPE i = 0 ; i < stride0 ; ++i )
    {
      dest[i] = source[i];
    }
  }

  /**
   * @brief function to copy values of another array to a location in this array
   * @param destIndex the index where the source data is to be inserted
   * @param source the source of the data
   *
   * This function will copy the data from source, to a location in *this. It is intended to allow
   * for collection of multiple data arrays into a single array. All dims should be the same except
   * for m_singleParameterResizeIndex;
   */
  void copy( INDEX_TYPE const destIndex, ArrayView const & source )
  {
    INDEX_TYPE offset = destIndex * m_strides[m_singleParameterResizeIndex];

    GEOS_ERROR_IF( source.size() > this->size() - offset,
                   "Insufficient storage space to copy source (size="<<source.size()<<
                   ") into current array at specified offset ("<<offset<<")."<<
                   " Available space is equal to this->size() - offset = "<<this->size() - offset );
    for( INDEX_TYPE i=0 ; i<source.size() ; ++i )
    {
      m_data[offset+i] = source.data()[i];
    }
  }

  /**
   * @brief function to get the length of a dimension
   * @param dim the dimension for which to get the length of
   */
  LVARRAY_HOST_DEVICE inline INDEX_TYPE size( int dim ) const noexcept
  {
    return m_dims[dim];
  }

  /**
   * @brief this function is an accessor for the dims array
   * @return a pointer to m_dims
   */
  LVARRAY_HOST_DEVICE inline INDEX_TYPE const * dims() const noexcept
  {
    return m_dims;
  }

  /**
   * @brief this function is an accessor for the strides array
   * @return a pointer to m_strides
   */
  inline INDEX_TYPE const * strides() const noexcept
  {
    return m_strides;
  }

  /**
   * @brief This function outputs the contents of an array to an output stream
   * @param stream the output stream for which to apply operator<<
   * @param array the array to output
   * @return a reference to the ostream
   */
  friend std::ostream& operator<< ( std::ostream& stream, ArrayView const & array )
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

  /**
   * Sets the ArraySlice::m_data pointer to the current contents of m_dataVector. This is not in
   * ArraySlice because we do not want the array slices data members to be immutable from within
   * the slice.
   */
  LVARRAY_HOST_DEVICE void setDataPtr() noexcept
  {
    T*& dataPtr = const_cast<T*&>(this->m_data);
    dataPtr = m_dataVector.data();
  }

  /**
   * @brief this function sets the dimensions of the array, but does not perform a resize to the
   *        new dimensions
   * @param dims a pointer/array containing the the new dimensions
   */
  template< typename DIMS_TYPE >
  LVARRAY_HOST_DEVICE void setDims( DIMS_TYPE const dims[NDIM] ) noexcept
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
#ifdef __CUDA_ARCH__
      const_cast<INDEX_TYPE *>(m_dims)[a] = dims[a];
#else
      const_cast<INDEX_TYPE *>(m_dims)[a] = integer_conversion<INDEX_TYPE>( dims[a] );
#endif
    }
  }

  /**
   * @brief this function sets the strides of the array.
   * @param strides a pointer/array containing the the new strides
   */
  LVARRAY_HOST_DEVICE void setStrides( INDEX_TYPE const strides[NDIM] ) noexcept
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      const_cast<INDEX_TYPE *>(m_strides)[a] = strides[a];
    }
  }


#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( ArrayView const * av )
  {
    if( av!=nullptr )
    {
      int constexpr ndim = NDIM;
      int errchk = 0;
      std::cout<<"Totalview using ("<<totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims )<<") for display of m_data;"<<std::endl;
      if ((errchk = TV_ttf_add_row("tv(m_data)", totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims ).c_str(), (av->m_data) )) != 0)
	std::cout<<"add_row returned error "<<errchk<<std::endl;
      if ((errchk = TV_ttf_add_row("m_data", totalview::format<T,INDEX_TYPE>(0, av->m_dims ).c_str(), (av->m_data) )) != 0)
        std::cout<<"add_row returned error "<<errchk<<std::endl;
      if ((errchk = TV_ttf_add_row("m_dims", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_dims) )) != 0 )
	std::cout<<"add_row returned error "<<errchk<<std::endl;
      if ((errchk = TV_ttf_add_row("m_strides", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_strides) )) != 0 )
	std::cout<<"add_row returned error "<<errchk<<std::endl;	
      if ((errchk = TV_ttf_add_row("m_dataVector", totalview::typeName<DATA_VECTOR_TYPE>().c_str() , &(av->m_dataVector) )) != 0 )
	std::cout<<"add_row returned error "<<errchk<<std::endl;	
      if ((errchk = TV_ttf_add_row("m_singleParameterResizeIndex", "int" , &(av->m_singleParameterResizeIndex) )) != 0 )
	std::cout<<"add_row returned error "<<errchk<<std::endl;      
    }
    return 0;
  }
#endif

protected:

  T * const restrict m_data;

  /// this data member contains the dimensions of the array
  INDEX_TYPE const m_dims[NDIM];

  /// this data member contains the strides of the array
  INDEX_TYPE const m_strides[NDIM];

  /// this data member contains the actual data for the array
  DATA_VECTOR_TYPE m_dataVector;

  /// this data member specifies the index that will be resized as a result of a call to the
  /// single argument version of the function resize(a)
  int m_singleParameterResizeIndex = 0;
};



} /* namespace LvArray */

#endif /* ARRAYVIEW_HPP_ */
