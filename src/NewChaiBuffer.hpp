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
#include "StringUtilities.hpp"
#include "bufferManipulation.hpp"

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

// CHAI is not threadsafe so we use a lock to serialize access.
static std::mutex chaiLock;

inline chai::ExecutionSpace toChaiExecutionSpace( MemorySpace const space )
{
  if( space == MemorySpace::NONE )
    return chai::NONE;
  if( space == MemorySpace::CPU )
    return chai::CPU;
#if defined(USE_CUDA)
  if( space == MemorySpace::GPU )
    return chai::GPU;
#endif

  LVARRAY_ERROR( "Unrecognized memory space " << static_cast< int >( space ) );

  return chai::NONE;
}

inline MemorySpace toMemorySpace( chai::ExecutionSpace const space )
{
  if( space == chai::NONE )
    return MemorySpace::NONE;
  if( space == chai::CPU )
    return MemorySpace::CPU;
#if defined(USE_CUDA)
  if( space == chai::GPU )
    return MemorySpace::GPU;
#endif

  LVARRAY_ERROR( "Unrecognized execution space " << static_cast< int >( space ) );

  return MemorySpace::NONE;
}

} // namespace internal

/**
 * @tparam T type of data that is contained in the buffer.
 * @class NewChaiBuffer
 * @brief Implements the Buffer interface using CHAI.
 * @details The NewChaiBuffer's allocation can exist in multiple memory spaces. If the chai
 *   execution space is set the copy constructor will ensure that the newly constructed
 *   NewChaiBuffer's pointer points to memory in that space. If the memory does
 *   exist it will be allocated and the data copied over. If the memory exists but the data has been
 *   touched (modified) in the current space it will be copied over. The data is touched in the
 *   new space if T is non const and is not touched if T is const.
 * @note Both the copy constructor and copy assignment constructor perform a shallow copy
 *   of the source. Similarly the destructor does not free the allocation.
 */
template< typename T >
class NewChaiBuffer
{
public:

  /// Alias for T used used in the bufferManipulation functions.
  using value_type = T;

  /// A flag indicating that the NewChaiBuffer's copy semantics are shallow.
  constexpr static bool hasShallowCopy = true;

  /// An alias for the non const version of T.
  using T_non_const = std::remove_const_t< T >;

  /**
   * @brief Default constructor, creates an uninitialized NewChaiBuffer.
   * @details An uninitialized NewChaiBuffer is an undefined state and may only be assigned to.
   *   An uninitialized NewChaiBuffer holds no recources and does not need to be free'd.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  NewChaiBuffer():
    m_pointer( nullptr ),
    m_capacity( 0 ),
    m_pointer_record( nullptr )
  {}

  /**
   * @brief Constructor for creating an empty Buffer.
   * @details An empty buffer may hold resources and needs to be free'd.
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
      m_pointer_record->m_allocators[ space ] = internal::getArrayManager().getAllocatorId( chai::ExecutionSpace( space ) );
    }
  }

  /**
   * @brief Copy constructor.
   * @param src The buffer to copy.
   * @details In addition to performing a shallow copy of @p src if the chai execution space
   *   is set *this will contain a pointer the the allocation in that space.
   */
  LVARRAY_HOST_DEVICE inline
  NewChaiBuffer( NewChaiBuffer const & src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointer_record( src.m_pointer_record )
  {
  #if defined(USE_CUDA) && !defined(__CUDA_ARCH__)
    move( internal::toMemorySpace( internal::getArrayManager().getExecutionSpace() ), true );
  #endif
  }

  /**
   * @copydoc NewChaiBuffer( NewChaiBuffer const & )
   * @param size The number of values in the allocation.
   * @note This method should be preffered over the copy constructor when the size information
   *   is available.
   */
  LVARRAY_HOST_DEVICE inline
  NewChaiBuffer( NewChaiBuffer const & src, std::ptrdiff_t const size ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointer_record( src.m_pointer_record )
  {
  #if defined(USE_CUDA) && !defined(__CUDA_ARCH__)
    move( internal::toMemorySpace( internal::getArrayManager().getExecutionSpace() ), size, true );
  #else
    LVARRAY_UNUSED_VARIABLE( size );
  #endif
  }

  /**
   * @brief Move constructor.
   * @param src The NewChaiBuffer to be moved from, is uninitialized after this call.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  NewChaiBuffer( NewChaiBuffer && src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointer_record( src.m_pointer_record )
  {
    src.m_capacity = 0;
    src.m_pointer = nullptr;
    src.m_pointer_record = nullptr;
  }

  /**
   * @brief Copy assignment operator.
   * @param src The NewChaiBuffer to be copied.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  NewChaiBuffer & operator=( NewChaiBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_pointer = src.m_pointer;
    m_pointer_record = src.m_pointer_record;
    return *this;
  }

  /**
   * @brief Move assignment operator.
   * @param src The NewChaiBuffer to be moved from, is uninitialized after this call.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline constexpr
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
   *   Values between [0, size) are destroyed.
   * @param newCapacity the new capacity of the buffer.
   * @note This currently only reallocates the buffer on the CPU and it frees
   *   the buffer in every other memory space.
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
    registerTouch( MemorySpace::CPU );
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note To destroy the values and free the data call bufferManipulation::free.
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
   * @brief @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  std::ptrdiff_t capacity() const
  { return m_capacity; }

  /**
   * @brief @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_pointer; }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @brief @return The value at position @p i .
   * @param i The position of the value to access.
   * @note No bounds checks are performed.
   */
  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator[]( INDEX_TYPE const i ) const
  { return m_pointer[ i ]; }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space The space to move the buffer to.
   * @param size The size of the buffer.
   * @param touch If the buffer should be touched in the new space or not.
   */
  inline
  void move( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
  #if defined(USE_CUDA)
    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );
    if( m_pointer_record == nullptr ||
        m_capacity == 0 ||
        chaiSpace == chai::NONE ) return;

    chai::ExecutionSpace const prevSpace = m_pointer_record->m_last_space;

    if( prevSpace == chai::CPU && prevSpace != chaiSpace ) moveInnerData( space, size, touch );

    const_cast< T * & >( m_pointer ) =
      static_cast< T * >( internal::getArrayManager().move( const_cast< T_non_const * >( m_pointer ),
                                                            m_pointer_record,
                                                            chaiSpace ) );

    if( !std::is_const< T >::value && touch ) m_pointer_record->m_touched[ chaiSpace ] = true;
    m_pointer_record->m_last_space = chaiSpace;

    if( prevSpace == chai::GPU && prevSpace != chaiSpace ) moveInnerData( space, size, touch );

  #else
    LVARRAY_ERROR_IF_NE( space, MemorySpace::CPU );
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_UNUSED_VARIABLE( touch );
  #endif
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space The space to move the buffer to.
   * @param touch If the buffer should be touched in the new space or not.
   * @note This method is only active when the type T itself does not have a method move( MemorySpace ).
   * @return Nothing.
   */
  template< typename U=T_non_const >
  std::enable_if_t< !bufferManipulation::HasMemberFunction_move< U > >
  move( MemorySpace const space, bool const touch ) const
  { move( space, capacity(), touch ); }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   */
  inline constexpr
  void registerTouch( MemorySpace const space ) const
  {
    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );
    m_pointer_record->m_touched[ chaiSpace ] = true;
    m_pointer_record->m_last_space = chaiSpace;
  }

  /**
   * @tparam U The type of the owning class, will be displayed in the callback.
   * @brief Set the name associated with this buffer which is used in the chai callback.
   * @param name the of the buffer.
   */
  template< typename U=NewChaiBuffer< T > >
  void setName( std::string const & name )
  {
    std::string const typeString = LvArray::demangle( typeid( U ).name() );
    m_pointer_record->m_user_callback =
      [name, typeString]( chai::PointerRecord const * const record, chai::Action const act, chai::ExecutionSpace const s )
    {
      if( act == chai::ACTION_MOVE )
      {
        std::string const size = LvArray::calculateSize( record->m_size );
        std::string const paddedSize = std::string( 9 - size.size(), ' ' ) + size;
        char const * const spaceStr = ( s == chai::CPU ) ? "HOST  " : "DEVICE";
        LVARRAY_LOG( "Moved " << paddedSize << " to the " << spaceStr << ": " << typeString << " " << name );
      }
    };
  }

private:

  /**
   * @tparam U A dummy parameter to enable SFINAE, do not specify.
   * @brief Move inner allocations to the memory space @p space.
   * @param space The memory space to move to.
   * @param size The number of values to move.
   * @param touch If the inner values should be touched or not.
   * @note This method is only active when T has a method move( MemorySpace ).
   */
  template< typename U=T_non_const >
  std::enable_if_t< bufferManipulation::HasMemberFunction_move< U > >
  moveInnerData( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
    if( space == MemorySpace::NONE ) return;

    for( std::ptrdiff_t i = 0; i < size; ++i )
    {
      const_cast< T_non_const * >( m_pointer )[ i ].move( space, touch );
    }
  }

  /**
   * @tparam U A dummy parameter to enable SFINAE, do not specify.
   * @brief Move inner allocations to the memory space @p space.
   * @note This method is only active when T does not have a method move( MemorySpace ).
   */
  template< typename U=T_non_const >
  std::enable_if_t< !bufferManipulation::HasMemberFunction_move< U > >
  moveInnerData( MemorySpace const, std::ptrdiff_t const, bool const ) const
  {}

  /// A pointer to the data.
  T * LVARRAY_RESTRICT m_pointer = nullptr;

  /// The size of the allocation.
  std::ptrdiff_t m_capacity = 0;

  /// A pointer to the chai PointerRecord, keeps track of the memory space information.
  chai::PointerRecord * m_pointer_record = nullptr;
};

} /* namespace LvArray */
