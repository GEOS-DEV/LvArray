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

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"
#include "bufferManipulation.hpp"

// TPL includes
#include <stddef.h>


namespace LvArray
{

/**
 * @class MallocBuffer
 * @brief Implements the Buffer interface using malloc and free.
 * @tparam T type of data that is contained in the buffer.
 * @details Both the copy constructor and copy assignment constructor perform a shallow copy
 *   of the source. Similarly the destructor does not free the allocation.
 * @note The parent class bufferManipulation::VoidBuffer provides the default execution space related methods.
 */
template< typename T >
class MallocBuffer : public bufferManipulation::VoidBuffer
{
public:

  /// Alias used in the bufferManipulation functions.
  using value_type = T;

  /// Signifies that the MallocBuffer's copy semantics are shallow.
  static constexpr bool hasShallowCopy = true;

  /**
   * @brief Constructor for creating an empty or uninitialized buffer.
   * @note An uninitialized MallocBuffer is equivalent to an empty MallocBuffer and does
   *   not need to be free'd.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer( bool=true ):
    m_data( nullptr ),
    m_capacity( 0 )
  {}

  /**
   * @brief Copy constructor, creates a shallow copy.
   */
  MallocBuffer( MallocBuffer const & ) = default;

  /**
   * @brief Sized copy constructor, creates a shallow copy.
   * @param src The buffer to be coppied.
   */
  MallocBuffer( MallocBuffer const & src, std::ptrdiff_t ):
    MallocBuffer( src )
  {}

  /**
   * @brief Move constructor, creates a shallow copy.
   * @param src The buffer to be moved from, is empty after the move.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer( MallocBuffer && src ):
    m_data( src.m_data ),
    m_capacity( src.m_capacity )
  {
    src.m_capacity = 0;
    src.m_data = nullptr;
  }

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param src The buffer to be copied.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer & operator=( MallocBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_data = src.m_data;
    return *this;
  }

  /**
   * @brief Move assignment operator, creates a shallow copy.
   * @param src The buffer to be moved from, is empty after the move.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline constexpr
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
   * @param size The number of values that are initialized in the buffer.
   * @param newCapacity The new capacity of the buffer.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    // TODO: If std::is_trivially_copyable_v< T > then we could use std::realloc.
    T * const newPtr = reinterpret_cast< T * >( std::malloc( newCapacity * sizeof( T ) ) );

    std::ptrdiff_t const overlapAmount = std::min( newCapacity, size );
    arrayManipulation::uninitializedMove( newPtr, overlapAmount, m_data );
    arrayManipulation::destroy( m_data, size );

    std::free( m_data );
    m_capacity = newCapacity;
    m_data = newPtr;
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note To destroy the values and free the data call bufferManipulation::free.
   */
  LVARRAY_HOST_DEVICE inline
  void free()
  {
    std::free( m_data );
    m_capacity = 0;
    m_data = nullptr;
  }

  /**
   * @brief @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline
  std::ptrdiff_t capacity() const
  { return m_capacity; }

  /**
   * @brief @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_data; }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @brief @return The value at position @p i .
   * @param i The position of the value to access.
   * @note No bounds checks are performed.
   */
  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator[]( INDEX_TYPE const i ) const
  { return m_data[ i ]; }

private:

  /// A pointer to the data.
  T * LVARRAY_RESTRICT m_data = nullptr;

  /// The size of the allocation.
  std::ptrdiff_t m_capacity = 0;
};

} // namespace LvArray
