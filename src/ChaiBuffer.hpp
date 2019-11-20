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

#ifndef CHAI_BUFFER_HPP
#define CHAI_BUFFER_HPP

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "arrayManipulation.hpp"
#include "StringUtilities.hpp"

// TPL includes
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"

// System includes
#include <mutex>


namespace LvArray
{

namespace internal
{

// CHAI is not threadsafe so we use a lock to serialize access.
static std::mutex chaiLock;

/**
 * @brief Return a string representing the given size in bytes converted to either
 *        KB or MB.
 */
inline std::string calculateSize( size_t const bytes )
{
  if( bytes >> 20 != 0 )
  {
    return std::to_string( bytes >> 20 ) + "MB";
  }
  else
  {
    return std::to_string( bytes >> 10 ) + "KB";
  }
}

} // namespace internal

/**
 * @class ChaiBuffer
 * @brief This class implements the Buffer interface using CHAI and allows classes built on top
 *        of it to exist in multiple memory spaces.
 * @tparam T type of data that is contained in the buffer.
 * @note Both the copy constructor and copy assignment constructor perform a shallow copy
 *       of the source. Similarly the destructor does not free the allocation. This is
 *       the standard behavior of the Buffer classes.
 * @note The parent class chai::CHAICopyable allows for the movement of nested ChaiBuffers.
 */
template< typename T >
class ChaiBuffer : public chai::CHAICopyable
{
public:

  // Alias used in the bufferManipulation functions.
  using value_type = T;

  /**
   * @brief Default constructor. Creates an uninitialized Buffer. An uninitialized
   *        buffer is an undefined state and may only be assigned to. An uninitialized
   *        buffer holds no recources and does not need to be free'd.
   */
  ChaiBuffer():
    m_array( nullptr )
  {}

  /**
   * @brief Constructor for creating an empty Buffer. An empty buffer may hold resources
   *        and needs to be free'd.
   * @note The unused boolean parameter is to distinguish this from default constructor.
   */
  ChaiBuffer( bool ):
    m_array()
  {
#if !defined(__CUDA_ARCH__)
    setName( "" );
#endif
  }

  /**
   * @brief Reallocate the buffer to the new capacity.
   * @param size the number of values that are initialized in the buffer.
   *        values between [0, size) are destroyed.
   * @param newCapacity the new capacity of the buffer.
   * @note This currently only reallocates the buffer on the CPU and it frees
   *       the buffer in every other memory space.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    internal::chaiLock.lock();
    chai::ManagedArray< T > newArray( newCapacity );
    internal::chaiLock.unlock();

#if defined(USE_CUDA)
    newArray.setUserCallback( m_array.getPointerRecord()->m_user_callback );
#endif

    std::ptrdiff_t const overlapAmount = std::min( newCapacity, size );
    arrayManipulation::uninitializedMove( &newArray[ 0 ], overlapAmount, data() );
    arrayManipulation::destroy( data(), size );

    free();
    m_array = std::move( newArray );
    registerTouch( chai::CPU );
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values. To
   *        properly destroy the values and free the data call bufferManipulation::free.
   */
  inline
  void free()
  {
    std::lock_guard< std::mutex > lock( internal::chaiLock );
    m_array.free();
    m_array = nullptr;
  }

  /**
   * @brief Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::ptrdiff_t capacity() const
  {
    return m_array.size();
  }

  /**
   * @brief Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  {
    return &m_array[0];
  }

  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator[]( INDEX_TYPE const i ) const
  {
    return m_array[ i ];
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   */
  void move( chai::ExecutionSpace const space, bool const touch )
  {
#if defined(USE_CUDA)
    if( capacity() == 0 ) return;

    void * ptr = const_cast< typename std::remove_const< T >::type * >( data() );
    if( space == chai::ArrayManager::getInstance()->getPointerRecord( ptr )->m_last_space ) return;

    if( touch ) m_array.move( space );
    else reinterpret_cast< chai::ManagedArray< T const > & >( m_array ).move( space );
#else
    CXX_UTILS_UNUSED_VARIABLE( space );
    CXX_UTILS_UNUSED_VARIABLE( touch );
#endif
  }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   */
  void registerTouch( chai::ExecutionSpace const space )
  {
    m_array.registerTouch( space );
  }

  /**
   * @brief Set the name associated with this buffer which is used in the chai callback.
   * @param name the of the buffer.
   */
  template< typename U=ChaiBuffer< T > >
  void setName( std::string const & name )
  {
#if defined(USE_CUDA)
    std::string const typeString = cxx_utilities::demangle( typeid( U ).name() );
    m_array.setUserCallback( [name, typeString]( chai::Action act, chai::ExecutionSpace s, size_t bytes )
      {
        if( act == chai::ACTION_MOVE )
        {
          std::string const & size = internal::calculateSize( bytes );
          char const * const spaceStr = ( s == chai::CPU ) ? "CPU" : "GPU";
          LVARRAY_LOG( "Moved " << size << " to the " << spaceStr << ": " << typeString << " " << name );
        }
      } );
#else
    CXX_UTILS_UNUSED_VARIABLE( name );
#endif
  }

private:
  chai::ManagedArray< T > m_array;
};

} /* namespace LvArray */

#endif /* CHAI_BUFFER_HPP */
