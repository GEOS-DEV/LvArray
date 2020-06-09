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
#include "templateHelpers.hpp"
#include "arrayManipulation.hpp"

// System includes
#include <utility>

namespace LvArray
{

/**
 * @enum MemorySpace
 * @brief An enum containing the available memory spaces.
 */
enum class MemorySpace
{
  NONE
  , CPU
#if defined(USE_CUDA)
  , GPU
#endif
};

/**
 * @brief Output a MemorySpace enum to a stream.
 * @param os The output stream to write to.
 * @param space The MemorySpace to output.
 * @return @p os.
 */
inline std::ostream & operator<<( std::ostream & os, MemorySpace const space )
{
  if( space == MemorySpace::NONE )
    os << "NONE";
  if( space == MemorySpace::CPU )
    os << "CPU";
#if defined(USE_CUDA)
  if( space == MemorySpace::GPU )
    os << "GPU";
#endif
  return os;
}

namespace bufferManipulation
{

/**
 * @brief Defines a static constexpr bool HasMemberFunction_move< @p CLASS >
 *   that is true iff the method @p CLASS ::move(MemorySpace, bool) exists.
 * @tparam CLASS The type to test.
 */
IS_VALID_EXPRESSION( HasMemberFunction_move, CLASS, std::declval< CLASS >().move( MemorySpace::CPU, true ) );

/**
 * @class VoidBuffer
 * @brief This class implements the default behavior for the Buffer methods related
 *   to execution space. This class is not intended to be used directly, instead derive
 *   from it if you would like to inherit the default behavior.
 */
struct VoidBuffer
{
  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space the space to move the buffer to.
   * @param size the size of the buffer.
   * @param touch whether the buffer should be touched in the new space or not.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void move( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_UNUSED_VARIABLE( touch );
    LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::CPU, "This Buffer type can only be used on the CPU." );
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space the space to move the buffer to.
   * @param touch whether the buffer should be touched in the new space or not.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void move( MemorySpace const space, bool const touch ) const
  {
    LVARRAY_UNUSED_VARIABLE( touch );
    LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::CPU, "This Buffer type can only be used on the CPU." );
  }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void registerTouch( MemorySpace const space ) const
  { LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::CPU, "This Buffer type can only be used on the CPU." ); }

  /**
   * @tparam The type of the owning object.
   * @brief Set the name associated with this buffer.
   * @param name the name of the buffer.
   */
  template< typename=VoidBuffer >
  LVARRAY_HOST_DEVICE
  void setName( std::string const & name )
  { LVARRAY_UNUSED_VARIABLE( name ); }
};

/**
 * @brief Check that given Buffer and size are valid.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to check.
 * @param size the size of the buffer.
 * @note This method is a no-op when USE_ARRAY_BOUNDS_CHECK is not defined.
 */
template< typename BUFFER >
inline LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
void check( BUFFER const & buf, std::ptrdiff_t const size )
{
#ifdef USE_ARRAY_BOUNDS_CHECK
  LVARRAY_ERROR_IF_GT( 0, buf.capacity() );
  LVARRAY_ERROR_IF_GT( 0, size );
  LVARRAY_ERROR_IF_GT( size, buf.capacity() );
#else
  LVARRAY_DEBUG_VAR( buf );
  LVARRAY_DEBUG_VAR( size );
#endif
}

/**
 * @brief Check that given Buffer, size, and insertion position, are valid.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to check.
 * @param size the size of the buffer.
 * @param pos the insertion position.
 * @note This method is a no-op when USE_ARRAY_BOUNDS_CHECK is not defined.
 */
template< typename BUFFER >
inline
void checkInsert( BUFFER const & buf, std::ptrdiff_t const size, std::ptrdiff_t const pos )
{
#ifdef USE_ARRAY_BOUNDS_CHECK
  check( buf, size );
  LVARRAY_ERROR_IF_GT( 0, pos );
  LVARRAY_ERROR_IF_GT( pos, size );
#else
  LVARRAY_DEBUG_VAR( buf );
  LVARRAY_DEBUG_VAR( size );
  LVARRAY_DEBUG_VAR( pos );
#endif
}

/**
 * @brief Destroy the values in the buffer and free it's memory.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to free.
 * @param size the size of the buffer.
 */
DISABLE_HD_WARNING
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void free( BUFFER & buf, std::ptrdiff_t const size )
{
  using T = typename BUFFER::value_type;

  check( buf, size );

  /// If the values contained can themselves be moved then they need to be moved back to the host.
#if !defined(__CUDA_ARCH__ )
  if( HasMemberFunction_move< T > )
  {
    buf.move( MemorySpace::CPU, size, true );
  }
#endif

  T * const LVARRAY_RESTRICT data = buf.data();
  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    data[ i ].~T();
  }

  buf.free();
}

/**
 * @brief Set the capacity of the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to set the capacity of.
 * @param size the size of the buffer.
 * @param newCapacity the new capacity of the buffer.
 */
DISABLE_HD_WARNING
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void setCapacity( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
{
  check( buf, size );
  buf.reallocate( size, newCapacity );
}

/**
 * @brief Reserve space in the buffer for at least the given capacity.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to reserve space in.
 * @param size the size of the buffer.
 * @param newCapacity the new minimum capacity of the buffer.
 */
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void reserve( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
{
  check( buf, size );

  if( newCapacity > buf.capacity() )
  {
    setCapacity( buf, size, newCapacity );
  }
}

/**
 * @brief If the buffer's capacity is greater than newCapacity this is a no-op.
 *   Otherwise the buffer's capacity is increased to at least 2 * newCapacity.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to reserve space in.
 * @param size the size of the buffer.
 * @param newCapacity the new minimum capacity of the buffer.
 * @note Use this in methods which increase the size of the buffer to efficiently grow
 *   the capacity.
 */
template< typename BUFFER >
void dynamicReserve( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
{
  check( buf, size );

  if( newCapacity > buf.capacity() )
  {
    setCapacity( buf, size, 2 * newCapacity );
  }
}

/**
 * @brief Resize the buffer to the given size.
 * @tparam BUFFER the buffer type.
 * @param ARGS the types of the arguments to initialize the new values with.
 * @param buf the buffer to resize.
 * @param size the current size of the buffer.
 * @param newSize the new size of the buffer.
 * @param args the arguments to initialize the new values with.
 * @note Use this in methods which increase the size of the buffer to efficiently grow
 *   the capacity.
 */
template< typename BUFFER, typename ... ARGS >
LVARRAY_HOST_DEVICE
void resize( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newSize, ARGS && ... args )
{
  check( buf, size );

  reserve( buf, size, newSize );

  arrayManipulation::resize( buf.data(), size, newSize, std::forward< ARGS >( args )... );

#if !defined(__CUDA_ARCH__)
  if( newSize > 0 )
  {
    buf.registerTouch( MemorySpace::CPU );
  }
#endif
}


template< typename BUFFER, typename ... ARGS >
void emplaceBack( BUFFER & buf, std::ptrdiff_t const size, ARGS && ... args )
{
  check( buf, size );

  dynamicReserve( buf, size, size + 1 );
  arrayManipulation::emplaceBack( buf.data(), size, std::forward< ARGS >( args ) ... );
}

template< typename BUFFER, typename ... ARGS >
void emplace( BUFFER & buf,
              std::ptrdiff_t const size,
              std::ptrdiff_t const pos,
              ARGS && ... args )
{
  checkInsert( buf, size, pos );

  dynamicReserve( buf, size, size + 1 );
  arrayManipulation::emplace( buf.data(), size, pos, std::forward< ARGS >( args ) ... );
}

/**
 * @brief Insert multiple values into the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to insert into.
 * @param size the current size of the buffer.
 * @param pos the position to insert the values at.
 * @param values a pointer to the values to insert via a copy.
 * @param nVals the number of values to insert.
 * @return The number of values inserted.
 */
template< typename BUFFER, typename ITERATOR >
std::ptrdiff_t insert( BUFFER & buf,
                       std::ptrdiff_t const size,
                       std::ptrdiff_t const pos,
                       ITERATOR const first,
                       ITERATOR const last )
{
  checkInsert( buf, size, pos );

  std::ptrdiff_t const nVals = iterDistance( first, last );
  dynamicReserve( buf, size, size + nVals );
  arrayManipulation::insert( buf.data(), size, pos, first, nVals );
  return nVals;
}

/**
 * @brief Remove a value from the end of the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to remove from.
 * @param size the current size of the buffer.
 */
template< typename BUFFER >
void popBack( BUFFER & buf, std::ptrdiff_t const size )
{
  check( buf, size );
  arrayManipulation::popBack( buf.data(), size );
}

/**
 * @brief Erase a value from the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf the buffer to erase from.
 * @param size the current size of the buffer.
 * @param pos the position to erase.
 */
template< typename BUFFER >
void erase( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const pos )
{
  check( buf, size );
  LVARRAY_ERROR_IF_GE( pos, size );

  arrayManipulation::erase( buf.data(), size, pos, std::ptrdiff_t( 1 ) );
}

/**
 * @brief Copy values from the source buffer into the destination buffer.
 * @tparam DST_BUFFER the destination buffer type.
 * @tparam SRC_BUFFER the source buffer type.
 * @param dst the destination buffer.
 * @param dstSize the size of the destination buffer.
 * @param src the source buffer.
 * @param srcSize the size of the source buffer.
 */
DISABLE_HD_WARNING
template< typename DST_BUFFER, typename SRC_BUFFER >
LVARRAY_HOST_DEVICE
void copyInto( DST_BUFFER & dst,
               std::ptrdiff_t const dstSize,
               SRC_BUFFER const & src,
               std::ptrdiff_t const srcSize )
{
  check( dst, dstSize );
  check( src, srcSize );

  resize( dst, dstSize, srcSize );

  using T = typename DST_BUFFER::value_type;
  T * const LVARRAY_RESTRICT dstData = dst.data();
  T const * const LVARRAY_RESTRICT srcData = src.data();

  for( std::ptrdiff_t i = 0; i < srcSize; ++i )
  {
    dstData[ i ] = srcData[ i ];
  }
}

} // namespace bufferManipulations
} // namespace LvArray
