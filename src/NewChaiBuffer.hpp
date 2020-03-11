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

#ifndef NEW_CHAI_BUFFER_HPP
#define NEW_CHAI_BUFFER_HPP

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "ChaiBuffer.hpp"
#include "arrayManipulation.hpp"
#include "StringUtilities.hpp"

// TPL includes
#include <chai/ArrayManager.hpp>

// System includes
#include <mutex>


namespace LvArray
{

namespace internal
{

inline chai::ArrayManager & getArrayManager()
{
  static chai::ArrayManager & arrayManager = *chai::ArrayManager::getInstance();
  return arrayManager;
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
class NewChaiBuffer : public chai::CHAICopyable
{
public:

  // Alias used in the bufferManipulation functions.
  using value_type = T;
  using T_non_const = std::remove_const_t< T >;
  constexpr static bool hasShallowCopy = true;

  /**
   * @brief Default constructor. Creates an uninitialized Buffer. An uninitialized
   *        buffer is an undefined state and may only be assigned to. An uninitialized
   *        buffer holds no recources and does not need to be free'd.
   */
  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  NewChaiBuffer():
    m_pointer( nullptr ),
    m_capacity( 0 ),
    m_pointer_record( nullptr )
  {}

  /**
   * @brief Constructor for creating an empty Buffer. An empty buffer may hold resources
   *        and needs to be free'd.
   * @note The unused boolean parameter is to distinguish this from default constructor.
   */
  NewChaiBuffer( bool ):
    m_pointer( nullptr ),
    m_capacity( 0 ),
    m_pointer_record( new chai::PointerRecord{} )
  {
    m_pointer_record->m_size = 0;
    setName( "" );

    for( int space = chai::CPU; space < chai::NUM_EXECUTION_SPACES; ++space )
    {
      m_pointer_record->m_allocators[ space ] = internal::getArrayManager().getAllocatorId( chai::ExecutionSpace( space ));
    }
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE
  NewChaiBuffer( NewChaiBuffer const & src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointer_record( src.m_pointer_record )
  {
  #if defined(USE_CUDA) && !defined(__CUDA_ARCH__)
    move( internal::getArrayManager().getExecutionSpace() );
  #endif
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  NewChaiBuffer( NewChaiBuffer && src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointer_record( src.m_pointer_record )
  {
    src.m_capacity = 0;
    src.m_pointer = nullptr;
    src.m_pointer_record = nullptr;
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  NewChaiBuffer & operator=( NewChaiBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_pointer = src.m_pointer;
    m_pointer_record = src.m_pointer_record;
    return *this;
  }

  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  NewChaiBuffer & operator=( NewChaiBuffer && src )
  {
    m_capacity = src.m_capacity;
    m_pointer = src.m_pointer;
    m_pointer_record = src.m_pointer_record;

    src.m_capacity = 0;
    src.m_pointer = nullptr;
    src.m_pointer_record = nullptr;

    return *this;
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
    chai::PointerRecord * const newRecord = new chai::PointerRecord{};
    newRecord->m_size = newCapacity * sizeof( T );
    newRecord->m_user_callback = m_pointer_record->m_user_callback;

    for( int space = chai::CPU; space < chai::NUM_EXECUTION_SPACES; ++space )
    {
      newRecord->m_allocators[ space ] = m_pointer_record->m_allocators[ space ];
    }

    internal::chaiLock.lock();
    internal::getArrayManager().allocate( newRecord, chai::CPU );
    internal::chaiLock.unlock();

    T * const newPointer = static_cast< T * >( newRecord->m_pointers[ chai::CPU ] );

    std::ptrdiff_t const overlapAmount = std::min( newCapacity, size );
    arrayManipulation::uninitializedMove( newPointer, overlapAmount, m_pointer );
    arrayManipulation::destroy( m_pointer, size );

    free();
    m_capacity = newCapacity;
    m_pointer = newPointer;
    m_pointer_record = newRecord;
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
    internal::getArrayManager().free( m_pointer_record );
    m_capacity = 0;
    m_pointer = nullptr;
    m_pointer_record = nullptr;
  }

  /**
   * @brief Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  std::ptrdiff_t capacity() const
  {
    return m_capacity;
  }

  /**
   * @brief Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  T * data() const
  {
    return m_pointer;
  }

  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE RAJA_INLINE constexpr
  T & operator[]( INDEX_TYPE const i ) const
  {
    return m_pointer[ i ];
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   */
  inline
  void move( chai::ExecutionSpace const space, bool const touch=true )
  {
  #if defined(USE_CUDA)
    if( m_pointer_record == nullptr ||
        m_capacity == 0 ||
        space == chai::NONE ) return;

    chai::ExecutionSpace const prevSpace = m_pointer_record->m_last_space;

    if( prevSpace == chai::CPU ) moveInnerData( space );

    m_pointer = static_cast< T * >( internal::getArrayManager().move( const_cast< T_non_const * >( m_pointer ),
                                                                      m_pointer_record,
                                                                      space ) );

    if( !std::is_const< T >::value && touch ) m_pointer_record->m_touched[ space ] = true;
    m_pointer_record->m_last_space = space;

    if( prevSpace == chai::GPU ) moveInnerData( space );

  #else
    LVARRAY_ERROR_IF_NE( space, chai::CPU );
    LVARRAY_UNUSED_VARIABLE( touch );
  #endif
  }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   * @note This is done manually instead of calling ManagedArray::registerTouch
   *       to inline the method and avoid the std::lock used in ArrayManager::registerTouch.
   *       This call therefore is not threadsafe, but that shouldn't be a problem.
   */
  RAJA_INLINE constexpr
  void registerTouch( chai::ExecutionSpace const space )
  {
    m_pointer_record->m_touched[ space ] = true;
    m_pointer_record->m_last_space = space;
  }

  /**
   * @brief Set the name associated with this buffer which is used in the chai callback.
   * @param name the of the buffer.
   */
  template< typename U=NewChaiBuffer< T > >
  void setName( std::string const & name )
  {
    std::string const typeString = cxx_utilities::demangle( typeid( U ).name() );
    m_pointer_record->m_user_callback = \
      [name, typeString]( chai::Action act, chai::ExecutionSpace s, size_t bytes )
    {
      if( act == chai::ACTION_MOVE )
      {
        std::string const & size = internal::calculateSize( bytes );
        char const * const spaceStr = ( s == chai::CPU ) ? "HOST  " : "DEVICE";
        LVARRAY_LOG( "Moved " << size << " to the " << spaceStr << ": " << typeString << " " << name );
      }
    };
  }

private:

  template< typename U=T >
  std::enable_if_t< !std::is_base_of< CHAICopyable, U >::value >
  moveInnerData( chai::ExecutionSpace const LVARRAY_UNUSED_ARG( space ) )
  {}

  template< typename U=T >
  std::enable_if_t< std::is_base_of< CHAICopyable, U >::value >
  moveInnerData( chai::ExecutionSpace const space )
  {
    if( space == chai::NONE ) return;

    chai::ExecutionSpace const prevSpace = internal::getArrayManager().getExecutionSpace();
    internal::getArrayManager().setExecutionSpace( space );

    for( std::ptrdiff_t i = 0; i < m_capacity; ++i )
    {
      const_cast< T_non_const * >( m_pointer )[ i ] = T( m_pointer[ i ] );
    }

    internal::getArrayManager().setExecutionSpace( prevSpace );
  }

  T * LVARRAY_RESTRICT m_pointer = nullptr;
  std::ptrdiff_t m_capacity = 0;
  chai::PointerRecord * m_pointer_record = nullptr;
};

} /* namespace LvArray */

#endif /* NEW_CHAI_BUFFER_HPP */
