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

#ifndef MALLOC_BUFFER_HPP_
#define MALLOC_BUFFER_HPP_

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "bufferManipulation.hpp"

// TPL includes
#include <stddef.h>


namespace LvArray
{

/**
 * @class MallocBuffer
 * @brief This class implements the Buffer interface using the standard library
 *        functions malloc and free. It also serves as a reference for the standard
 *        functionality of the Buffer classes.
 * @tparam T type of data that is contained in the buffer.
 * @note Both the copy constructor and copy assignment constructor perform a shallow copy
 *       of the source. Similarly the destructor does not free the allocation. This is
 *       the standard behavior of the Buffer classes.
 * @note The parent class provides the default execution space related methods.
 */
template< typename T >
class MallocBuffer : public bufferManipulation::VoidBuffer
{
public:

  // Alias used in the bufferManipulation functions.
  using value_type = T;

  // member that signifies that this Buffer implementation supports shallow copies.
  static constexpr bool hasShallowCopy = true;

  /**
   * @brief Constructor for creating an empty/uninitialized buffer. For the MallocBuffer
   *        an uninitialized buffer is equivalent to an empty buffer.
   */
  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  MallocBuffer( bool=true ):
    m_data( nullptr ),
    m_capacity( 0 )
  {}

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  MallocBuffer( MallocBuffer const & src ):
    m_data( src.m_data ),
    m_capacity( src.m_capacity )
  {}

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  MallocBuffer( MallocBuffer && src ):
    m_data( src.m_data ),
    m_capacity( src.m_capacity )
  {
    src.m_capacity = 0;
    src.m_data = nullptr;
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  MallocBuffer & operator=( MallocBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_data = src.m_data;
    return *this;
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  MallocBuffer & operator=( MallocBuffer && src )
  {
    m_capacity = src.m_capacity;
    m_data = src.m_data;
    src.m_capacity = 0;
    src.m_data = nullptr;
    return *this;
  }

  /**
   * @brief Reallocate the buffer to the new capacity.
   * @param size the number of values that are initialized in the buffer.
   *        values between [0, size) are destroyed.
   * @param newCapacity the new capacity of the buffer.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    // Note: If std::is_trivially_copyable_v< T > then we could use std::realloc.
    T * const newPtr = reinterpret_cast< T * >( std::malloc( newCapacity * sizeof( T ) ) );

    std::ptrdiff_t const overlapAmount = std::min( newCapacity, size );
    arrayManipulation::uninitializedMove( newPtr, overlapAmount, m_data );
    arrayManipulation::destroy( m_data, size );

    std::free( m_data );
    m_capacity = newCapacity;
    m_data = newPtr;
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values. To
   *        properly destroy the values and free the data call bufferManipulation::free.
   */
  LVARRAY_HOST_DEVICE inline
  void free()
  {
    std::free( m_data );
    m_capacity = 0;
    m_data = nullptr;
  }

  /**
   * @brief Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline
  std::ptrdiff_t capacity() const
  { return m_capacity; }

  /**
   * @brief Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  {
    return m_data;
  }

  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator[]( INDEX_TYPE const i ) const
  {
    return m_data[ i ];
  }

private:
  T * restrict m_data = nullptr;
  std::ptrdiff_t m_capacity = 0;
};

} // namespace LvArray

#endif // MALLOC_BUFFER_HPP_
