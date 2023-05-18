/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file bufferManipulation.hpp
 * @brief Contains functions for manipulating buffers.
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"
#include "typeManipulation.hpp"
#include "arrayManipulation.hpp"

// TPL includes
#include <camp/resource.hpp>

// System includes
#include <utility>

namespace LvArray
{

/// @brief an alias for camp::resources::Platform.
using MemorySpace = camp::resources::Platform;

/**
 * @brief Output a Platform enum to a stream.
 * @param os The output stream to write to.
 * @param space The MemorySpace to output.
 * @return @p os.
 */
inline std::ostream & operator<<( std::ostream & os, MemorySpace const space )
{
  if( space == MemorySpace::undefined )
    return os << "undefined";
  if( space == MemorySpace::host )
    return os << "host";
  if( space == MemorySpace::cuda )
    return os << "cuda";
  if( space == MemorySpace::omp_target )
    return os << "omp_target";
  if( space == MemorySpace::hip )
    return os << "hip";
  if( space == MemorySpace::sycl )
    return os << "sycl";

  LVARRAY_ERROR( "Unrecognized memory space " << static_cast< int >( space ) );
  return os;
}

/**
 * @brief Contains template functions for performing common operations on buffers.
 * @details Each function accepts a buffer and a size as the first two arguments.
 */
namespace bufferManipulation
{

/**
 * @brief Defines a static constexpr bool HasMemberFunction_move< @p CLASS >
 *   that is true iff the class has a method move(MemorySpace, bool).
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( move, MemorySpace::host, true );

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
   * @param space The space to move the buffer to.
   * @param size The size of the buffer.
   * @param touch Whether the buffer should be touched in the new space or not.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void moveNested( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_UNUSED_VARIABLE( touch );
    LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "This Buffer type can only be used on the CPU." );
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space The space to move the buffer to.
   * @param touch Whether the buffer should be touched in the new space or not.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void move( MemorySpace const space, bool const touch ) const
  {
    LVARRAY_UNUSED_VARIABLE( touch );
    LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "This Buffer type can only be used on the CPU." );
  }

  /**
   * @return The last space the buffer was moved to.
   * @note The default behavior is that the Buffer can only exist on the CPU.
   */
  MemorySpace getPreviousSpace() const
  { return MemorySpace::host; }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   * @note The default behavior is that the Buffer can only exist on the CPU and an error
   *   occurs if you try to move it to a different space.
   */
  void registerTouch( MemorySpace const space ) const
  { LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "This Buffer type can only be used on the CPU." ); }

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
 * @param buf The buffer to check.
 * @param size The size of the buffer.
 * @note This method is a no-op when LVARRAY_BOUNDS_CHECK is not defined.
 */
template< typename BUFFER >
inline LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
void check( BUFFER const & buf, std::ptrdiff_t const size )
{
#ifdef LVARRAY_BOUNDS_CHECK
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
 * @param buf The buffer to check.
 * @param size The size of the buffer.
 * @param pos The insertion position.
 * @note This method is a no-op when LVARRAY_BOUNDS_CHECK is not defined.
 */
template< typename BUFFER >
inline
void checkInsert( BUFFER const & buf, std::ptrdiff_t const size, std::ptrdiff_t const pos )
{
#ifdef LVARRAY_BOUNDS_CHECK
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
 * @param buf The buffer to free.
 * @param size The size of the buffer.
 */
DISABLE_HD_WARNING
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void free( BUFFER & buf, std::ptrdiff_t const size )
{
  using T = typename BUFFER::value_type;

  check( buf, size );

  if( !std::is_trivially_destructible< T >::value )
  {
    buf.move( MemorySpace::host, true );
    arrayManipulation::destroy( buf.data(), size );
  }

  buf.free();
}

/**
 * @brief Set the capacity of the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf The buffer to set the capacity of.
 * @param size The size of the buffer.
 * @param space The space to set the capacity in.
 * @param newCapacity The new capacity of the buffer.
 */
DISABLE_HD_WARNING
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void setCapacity( BUFFER & buf, std::ptrdiff_t const size, MemorySpace const space, std::ptrdiff_t const newCapacity )
{
  check( buf, size );
  buf.reallocate( size, space, newCapacity );
}

/**
 * @brief Reserve space in the buffer for at least the given capacity.
 * @tparam BUFFER the buffer type.
 * @param buf The buffer to reserve space in.
 * @param size The size of the buffer.
 * @param space The space to perform the reserve in.
 * @param newCapacity The new minimum capacity of the buffer.
 */
template< typename BUFFER >
LVARRAY_HOST_DEVICE
void reserve( BUFFER & buf, std::ptrdiff_t const size, MemorySpace const space, std::ptrdiff_t const newCapacity )
{
  check( buf, size );

  if( newCapacity > buf.capacity() )
  {
    setCapacity( buf, size, space, newCapacity );
  }
}

/**
 * @brief If the buffer's capacity is greater than newCapacity this is a no-op.
 *   Otherwise the buffer's capacity is increased to at least 2 * newCapacity.
 * @tparam BUFFER the buffer type.
 * @param buf The buffer to reserve space in.
 * @param size The size of the buffer.
 * @param newCapacity The new minimum capacity of the buffer.
 * @note Use this in methods which increase the size of the buffer to efficiently grow
 *   the capacity.
 */
template< typename BUFFER >
void dynamicReserve( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
{
  check( buf, size );

  if( newCapacity > buf.capacity() )
  {
    setCapacity( buf, size, MemorySpace::host, 2 * newCapacity );
  }
}

/**
 * @brief Resize the buffer to the given size.
 * @tparam BUFFER the buffer type.
 * @tparam ARGS The types of the arguments to initialize the new values with.
 * @param buf The buffer to resize.
 * @param size The current size of the buffer.
 * @param newSize The new size of the buffer.
 * @param args The arguments to initialize the new values with.
 */
template< typename BUFFER, typename ... ARGS >
LVARRAY_HOST_DEVICE
void resize( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newSize, ARGS && ... args )
{
  check( buf, size );

  reserve( buf, size, MemorySpace::host, newSize );

  arrayManipulation::resize( buf.data(), size, newSize, std::forward< ARGS >( args )... );

#if !defined(LVARRAY_DEVICE_COMPILE)
  if( newSize > 0 )
  {
    buf.registerTouch( MemorySpace::host );
  }
#endif
}

/**
 * @brief Construct a new value at the end of the buffer.
 * @tparam BUFFER The buffer type.
 * @tparam ARGS A variadic pack of argument types.
 * @param buf The buffer to insert into.
 * @param size The current size of the buffer.
 * @param args A variadic pack of parameters to construct the new value with.
 */
template< typename BUFFER, typename ... ARGS >
void emplaceBack( BUFFER & buf, std::ptrdiff_t const size, ARGS && ... args )
{
  check( buf, size );

  dynamicReserve( buf, size, size + 1 );
  arrayManipulation::emplaceBack( buf.data(), size, std::forward< ARGS >( args ) ... );
}

/**
 * @brief Construct a new value at position @p pos.
 * @tparam BUFFER The buffer type.
 * @tparam ARGS A variadic pack of argument types.
 * @param buf The buffer to insert into.
 * @param size The current size of the buffer.
 * @param pos The position to construct the values at.
 * @param args A variadic pack of parameters to construct the new value with.
 */
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
 * @tparam BUFFER The buffer type.
 * @tparam ITER An iterator type.
 * @param buf The buffer to insert into.
 * @param size The current size of the buffer.
 * @param pos The position to insert the values at.
 * @param first An iterator to the first value to insert.
 * @param last An iterator to the end of the values to insert.
 * @return The number of values inserted.
 */
template< typename BUFFER, typename ITER >
std::ptrdiff_t insert( BUFFER & buf,
                       std::ptrdiff_t const size,
                       std::ptrdiff_t const pos,
                       ITER const first,
                       ITER const last )
{
  checkInsert( buf, size, pos );

  std::ptrdiff_t const nVals = arrayManipulation::iterDistance( first, last );
  dynamicReserve( buf, size, size + nVals );
  arrayManipulation::insert( buf.data(), size, pos, first, nVals );
  return nVals;
}

/**
 * @brief Remove a value from the end of the buffer.
 * @tparam BUFFER the buffer type.
 * @param buf The buffer to remove from.
 * @param size The current size of the buffer.
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
 * @param buf The buffer to erase from.
 * @param size The current size of the buffer.
 * @param pos The position to erase.
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
 * @param dst The destination buffer.
 * @param dstSize The size of the destination buffer.
 * @param src The source buffer.
 * @param srcSize The size of the source buffer.
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
