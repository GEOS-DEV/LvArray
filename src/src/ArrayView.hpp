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

// Source includes
#include "Permutation.hpp"
#include "ArraySlice.hpp"
#include "Logger.hpp"
#include "helpers.hpp"
#include "IntegerConversion.hpp"
#include "SFINAE_Macros.hpp"

// System includes
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
#include <totalview/tv_helpers.hpp>
#include <totalview/tv_data_display.h>
#endif

namespace LvArray
{

/**
 * @class ArrayView
 * @brief This class serves to provide a "view" of a multidimensional array.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 * @tparam UNIT_STRIDE_DIM the dimension with a unit stride, in an Array with a standard layout
 *         this is the last dimension.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *         one template argument that describes the type of the data being stored (T).
 *
 * When using CHAI the copy copy constructor of this class calls the copy constructor for CHAI
 * which will move the data to the location of the touch (host or device). In general, the
 * ArrayView should be what is passed by value into a lambda that is used to launch kernels as it
 * copy will trigger the desired data motion onto the appropriate memory space.
 *
 * Key features:
 * 1) When using a ChaiBuffer as the BUFFER_TYPE the ArrayView copy constructor will move the data
 *    to the current execution space.
 * 2) Defines a slicing operator[].
 * 3) Defines operator() array accessor.
 * 3) operator[] and operator() are all const and may be called in non-mutable lambdas.
 * 4) Conversion operators to go from ArrayView<T> to ArrayView<T const>.
 * 5) Since the Array is derived from ArrayView, it may be upcasted:
 *      Array<T,NDIM> array;
 *      ArrayView<T,NDIM> const & arrView = array;
 *
 * A good guideline is to upcast to an ArrayView when you don't need allocation capabilities that
 * are only present in Array.
 */
template< typename T,
          int NDIM,
          int UNIT_STRIDE_DIM,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  static_assert( UNIT_STRIDE_DIM >= 0, "UNIT_STRIDE_DIM must be positive." );
  static_assert( UNIT_STRIDE_DIM < NDIM, "UNIT_STRIDE_DIM must be less than NDIM." );

  using ViewTypeConst = typename AsConstView< ArrayView< T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE, BUFFER_TYPE > >::type;

  using value_type = T;
  using iterator = T *;
  using const_iterator = T const *;
  using pointer = T *;
  using const_pointer = T const *;

  /**
   * @brief A constructor to create an uninitialized ArrayView. An uninitialized may
   *        not be used until it is assigned to.
   */
  ArrayView() = default;

  /**
   * @brief Copy Constructor
   * @param source the object to copy
   * @return *this
   *
   * @note The copy constructor will trigger the copy constructor for DATA_VECTOR_TYPE. When using
   * the ChaiBuffer this will trigger CHAI::ManagedArray copy construction, and thus may copy
   * the memory to the memory space associated with the copy. I.e. if passed into a
   * lambda which is used to execute a cuda kernel, the data will be allocated and copied
   * to the device (if it doesn't already exist).
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArrayView( ArrayView const & source ) noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer{ source.m_dataBuffer },
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    setDataPtr();
    setDims( source.m_dims );
    setStrides( source.m_strides );
  }

  /**
   * @brief Move constructor, creates a shallow copy and invalidates the source.
   * @param source object to move
   * @return *this
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArrayView( ArrayView && source ):
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer( std::move( source.m_dataBuffer ) ),
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    setDataPtr();
    setDims( source.m_dims );
    setStrides( source.m_strides );

    for ( int i = 0; i < NDIM; ++i )
    {
      const_cast< INDEX_TYPE * >( source.m_dims )[ i ] = 0;
      const_cast< INDEX_TYPE * >( source.m_strides )[ i ] = 0;
    }

    source.setDataPtr();
  }

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param rhs object to copy.
   * @return *this
   */
  inline CONSTEXPRFUNC
  ArrayView & operator=( ArrayView const & rhs ) noexcept
  {
    m_dataBuffer = rhs.m_dataBuffer;
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    setStrides( rhs.m_strides );
    setDims( rhs.m_dims );
    setDataPtr();
    return *this;
  }

  /**
   * @brief Move assignment operator, creates a shallow copy and invalidates the source.
   * @param rhs object to copy.
   * @return *this
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArrayView & operator=( ArrayView && rhs )
  {
    m_dataBuffer = std::move( rhs.m_dataBuffer );
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    setStrides( rhs.m_strides );
    setDims( rhs.m_dims );
    setDataPtr();
    return *this;
  }

  /**
   * @brief set all values of array to rhs.
   * @param rhs value that array will be set to.
   * @return *this.
   */
  DISABLE_HD_WARNING
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  std::enable_if_t< !(std::is_const<U>::value), ArrayView const & >
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

  /**
   * @brief Convert this ArrayView and any nested Array to ArrayView const.
   *        In addition the data held by the innermost array will be const.
   */
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  std::enable_if_t< !std::is_const<U>::value, ViewTypeConst & >
  toViewConst() const
  {
    return reinterpret_cast< ViewTypeConst & >( *this );
  }

  /**
   * @brief User Defined Conversion operator to move from an ArrayView<T> const to
   *        ArrayView<T const> const &. 
   * @note This is achieved by applying a reinterpret_cast to the this pointer,
   *       which is a safe operation as the only difference between the types is a
   *       const specifier on the template parameter T.
   */
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator std::enable_if_t< !std::is_const<U>::value,
                             ArrayView<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const & >
  () const noexcept
  {
    return reinterpret_cast<ArrayView<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const &>(*this);
  }

  /**
   * @brief Return an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArraySlice<T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE>
  toSlice() const noexcept
  {
    return ArraySlice<T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE>( m_data, m_dims, m_strides );
  }

  /**
   * @brief Return an immutable ArraySlice representing this ArrayView.
   */
  template< typename U=T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE>
  toSliceConst() const noexcept
  {
    return ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE>( m_data, m_dims, m_strides );
  }

  /**
   * @brief User defined conversion to an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator ArraySlice<T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE>() const noexcept
  {
    return toSlice();
  }

  /**
   * @brief User defined conversion to an immutable ArraySlice representing this ArrayView.
   */
  template< typename U=T >
  inline LVARRAY_HOST_DEVICE CONSTEXPRFUNC
  operator std::enable_if_t< !std::is_const<U>::value,
                             ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const >
  () const noexcept
  {
    return toSliceConst();
  }

  /**
   * @brief User defined conversion to convert a ArraySlice< T, 1, 0 > to a raw pointer.
   */
  template< int _NDIM=NDIM, int _UNIT_STRIDE_DIM=UNIT_STRIDE_DIM >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator std::enable_if_t< _NDIM == 1 && _UNIT_STRIDE_DIM == 0, T * const restrict >
  () const noexcept restrict_this
  {
    return m_data;
  }

  /**
   * @brief Return the allocated size.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE size() const noexcept
  { return multiplyAll< NDIM >( m_dims ); }

  /**
   * @brief Return the length of the given dimension.
   * @param dim the dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE size( int dim ) const noexcept
  {
#ifdef USE_ARRAY_BOUNDS_CHECK
    GEOS_ASSERT_GT( NDIM, dim );
#endif
    return m_dims[dim];
  }

  /**
   * @brief Return true if the array is empty.
   */
  inline bool empty() const
  { return size() == 0; }

  /**
   * @brief Return a pointer to the begining of the data.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * begin() const
  { return data(); }

  /**
   * @brief Return a pointer to the end of the data (one past the last value).
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * end() const
  { return data() + size(); }

  /**
   * @brief Return a reference to the first value.
   */
  T & front() const
  { return m_data[0]; }

  /**
   * @brief Return a reference to the last value.
   */
  T & back() const
  { return m_data[size() - 1]; }

  ///***********************************************************************************************
  ///**** DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************

  /**
   * @brief Return a lower dimensionsal slice of this ArrayView.
   * @param index the index of the slice to create.
   * @note This method is only active when NDIM > 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< (U > 1), ArraySlice< T, NDIM - 1, UNIT_STRIDE_DIM - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice<T, NDIM-1, UNIT_STRIDE_DIM-1, INDEX_TYPE>( m_data + ConditionalMultiply< 0, UNIT_STRIDE_DIM >::multiply( index, m_strides[ 0 ] ), m_dims + 1, m_strides + 1 );
  }

  /**
   * @brief Return a reference to the value at the given index.
   * @param index index of the element in array to access.
   * @note This method is only active when NDIM == 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< U == 1, T & >
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return m_data[ ConditionalMultiply< 0, UNIT_STRIDE_DIM >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @brief Return a reference to the value at the given multidimensional index.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request.
   * @note This is a standard fortran like parentheses interface to array access.
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_data[ linearIndex( indices... ) ];
  }

  /**
   * @brief Calculates the linear index from a multidimensional index.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request.
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    checkIndices( m_dims, indices... );
#endif
    return getLinearIndex< UNIT_STRIDE_DIM >( m_strides, indices... );
  }

  ///***********************************************************************************************
  ///**** END DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************  

  /**
   * @brief Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  { return m_data; }

  /**
   * @brief Return a pointer to a slice of the values.
   * @param index the index of the slice to get.
   * @todo THIS FUNCION NEEDS TO BE GENERALIZED for all dims.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data( INDEX_TYPE const index ) const
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return data() + index * m_strides[ 0 ];
  }

  /**
   * @brief Copy the values from one slice in this array to another.
   * @param destIndex index for which to copy data into.
   * @param sourceIndex index to copy data from.
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
   * @brief Return a pointer to the array containing the size of each dimension.
   */
  LVARRAY_HOST_DEVICE inline INDEX_TYPE const * dims() const noexcept
  {
    return m_dims;
  }

  /**
   * @brief Return a pointer to the array containing the stride of each dimension.
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
      //std::cout<<"Totalview using ("<<totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims )<<") for display of m_data;"<<std::endl;
      TV_ttf_add_row("tv(m_data)", totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row("m_data", totalview::format<T,INDEX_TYPE>(0, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row("m_dims", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_dims) );
      TV_ttf_add_row("m_strides", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_strides) );
      TV_ttf_add_row("m_dataBuffer", totalview::typeName<BUFFER_TYPE<T>>().c_str() , &(av->m_dataBuffer) );
      TV_ttf_add_row("m_singleParameterResizeIndex", "int" , &(av->m_singleParameterResizeIndex) );
    }
    return 0;
  }
#endif

protected:

  /**
   * @brief Protected constructor to be used by the Array class.
   * @note The unused boolean parameter is to distinguish this from the default constructor.
   */
  inline explicit CONSTEXPRFUNC
  ArrayView( bool ) noexcept:
    m_data{ nullptr },
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer( true )
  {
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
    ArrayView::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Sets the m_data pointer to the current contents of m_dataBuffer.
   */
  LVARRAY_HOST_DEVICE
  void setDataPtr() noexcept
  {
    T*& dataPtr = const_cast<T*&>(this->m_data);
    dataPtr = m_dataBuffer.data();
  }

  /**
   * @brief Sets the dimensions of the array without resizing.
   * @param dims a pointer to the new dimensions.
   */
  LVARRAY_HOST_DEVICE
  void setDims( INDEX_TYPE const dims[NDIM] ) noexcept
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      const_cast<INDEX_TYPE *>(m_dims)[a] = dims[a];
    }
  }

  /**
   * @brief Sets the strides of the array.
   * @param strides a pointer to the new strides
   */
  LVARRAY_HOST_DEVICE
  void setStrides( INDEX_TYPE const strides[NDIM] ) noexcept
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      const_cast<INDEX_TYPE *>(m_strides)[a] = strides[a];
    }
  }

  /// A pointer to the values.
  T * const restrict m_data = nullptr;

  /// the dimensions of the array.
  INDEX_TYPE const m_dims[NDIM] = { 0 };

  /// the strides of the array.
  INDEX_TYPE const m_strides[NDIM] = { 0 };

  /// this data member contains the actual data for the array.
  BUFFER_TYPE< T > m_dataBuffer;

  /// this data member specifies the dimension that will be resized as a result of a call to the
  /// single dimension resize method.
  int m_singleParameterResizeIndex = 0;
};

} /* namespace LvArray */

#endif /* ARRAYVIEW_HPP_ */
