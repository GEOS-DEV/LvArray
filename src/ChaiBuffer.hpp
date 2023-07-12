/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file ChaiBuffer.hpp
 * @brief Contains the implementation of LvArray::ChaiBuffer.
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"
#include "typeManipulation.hpp"
#include "arrayManipulation.hpp"
#include "system.hpp"
#include "bufferManipulation.hpp"

// TPL includes
#include <chai/ArrayManager.hpp>

// System includes
#include <mutex>


namespace LvArray
{

namespace internal
{

/**
 * @return The chai::ArrayManager instance.
 */
inline chai::ArrayManager & getArrayManager()
{
  static chai::ArrayManager & arrayManager = *chai::ArrayManager::getInstance();
  return arrayManager;
}

/// chai is not threadsafe so we use a lock to serialize access.
static std::mutex chaiLock;

/**
 * @return The chai::ExecutionSpace corresponding to @p space.
 * @param space The MemorySpace to convert.
 */
inline chai::ExecutionSpace toChaiExecutionSpace( MemorySpace const space )
{
  if( space == MemorySpace::undefined )
    return chai::NONE;
  if( space == MemorySpace::host )
    return chai::CPU;
#if defined(LVARRAY_USE_CUDA)
  if( space == MemorySpace::cuda )
    return chai::GPU;
#endif
#if defined(LVARRAY_USE_HIP)
  if( space == MemorySpace::hip )
    return chai::GPU;
#endif

  LVARRAY_ERROR( "Unrecognized memory space " << static_cast< int >( space ) );

  return chai::NONE;
}

/**
 * @return The MemorySpace corresponding to @p space.
 * @param space The chai::ExecutionSpace to convert.
 */
inline MemorySpace toMemorySpace( chai::ExecutionSpace const space )
{
  if( space == chai::NONE )
    return MemorySpace::undefined;
  if( space == chai::CPU )
    return MemorySpace::host;
#if defined(LVARRAY_USE_CUDA)
  if( space == chai::GPU )
    return MemorySpace::cuda;
#endif
#if defined(LVARRAY_USE_HIP)
  if( space == chai::GPU )
    return MemorySpace::hip;
#endif

  LVARRAY_ERROR( "Unrecognized execution space " << static_cast< int >( space ) );

  return MemorySpace::undefined;
}

} // namespace internal

/**
 * @tparam T type of data that is contained in the buffer.
 * @class ChaiBuffer
 * @brief Implements the Buffer interface using CHAI.
 * @details The ChaiBuffer's allocation can exist in multiple memory spaces. If the chai
 *   execution space is set the copy constructor will ensure that the newly constructed
 *   ChaiBuffer's pointer points to memory in that space. If the memory does
 *   exist it will be allocated and the data copied over. If the memory exists but the data has been
 *   touched (modified) in the current space it will be copied over. The data is touched in the
 *   new space if T is non const and is not touched if T is const.
 * @note Both the copy constructor and copy assignment constructor perform a shallow copy
 *   of the source. Similarly the destructor does not free the allocation.
 */
template< typename T >
class ChaiBuffer
{
public:

  /// Alias for T used used in the bufferManipulation functions.
  using value_type = T;

  /// A flag indicating that the ChaiBuffer's copy semantics are shallow.
  constexpr static bool hasShallowCopy = true;

  /// An alias for the non const version of T.
  using T_non_const = std::remove_const_t< T >;

  /**
   * @brief Default constructor, creates an uninitialized ChaiBuffer.
   * @details An uninitialized ChaiBuffer is an undefined state and may only be assigned to.
   *   An uninitialized ChaiBuffer holds no recources and does not need to be free'd.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  ChaiBuffer():
    m_pointer( nullptr ),
    m_capacity( 0 ),
    m_pointerRecord( nullptr )
  {}

  /**
   * @brief Constructor for creating an empty Buffer.
   * @details An empty buffer may hold resources and needs to be free'd.
   * @note The unused boolean parameter is to distinguish this from default constructor.
   * @note Although it is marked as a host-device method, this is only valid to call from the host.
   */
  LVARRAY_HOST_DEVICE
  ChaiBuffer( bool ):
    m_pointer( nullptr ),
    m_capacity( 0 )
  #if !defined(LVARRAY_DEVICE_COMPILE)
    , m_pointerRecord( new chai::PointerRecord{} )
  #else
    , m_pointerRecord( nullptr )
  #endif
  {
  #if defined(LVARRAY_DEVICE_COMPILE)
    LVARRAY_ERROR( "Creating a new ChaiBuffer on device is not supported. This is often the result of capturing an array on device instead of a view." );
  #else
    m_pointerRecord->m_size = 0;
    setName( "" );

    for( int space = chai::CPU; space < chai::NUM_EXECUTION_SPACES; ++space )
    {
      m_pointerRecord->m_allocators[ space ] = internal::getArrayManager().getAllocatorId( chai::ExecutionSpace( space ) );
    }
  #endif
  }

  /**
   * @brief Construct a ChaiBuffer which uses the specific allocator for each space.
   * @param spaces The list of spaces.
   * @param allocators The allocators, must be the same length as @p spaces.
   * @details @code allocator[ i ] @endcode is used for the memory space @code spaces[ i ] @endcode.
   * @note Although it is marked as a host-device method, this is only valid to call from the host.
   */
  LVARRAY_HOST_DEVICE
  ChaiBuffer( std::initializer_list< MemorySpace > const & spaces,
              std::initializer_list< umpire::Allocator > const & allocators ):
    m_pointer( nullptr ),
    m_capacity( 0 )
  #if !defined(LVARRAY_DEVICE_COMPILE)
    , m_pointerRecord( new chai::PointerRecord{} )
  #else
    , m_pointerRecord( nullptr )
  #endif
  {
  #if defined(LVARRAY_DEVICE_COMPILE)
    LVARRAY_ERROR( "Creating a new ChaiBuffer on device is not supported." );
  #else
    m_pointerRecord->m_size = 0;
    setName( "" );

    LVARRAY_ERROR_IF_NE( spaces.size(), allocators.size() );

    for( int space = chai::CPU; space < chai::NUM_EXECUTION_SPACES; ++space )
    {
      m_pointerRecord->m_allocators[ space ] = internal::getArrayManager().getAllocatorId( chai::ExecutionSpace( space ) );
    }

    for( std::size_t i = 0; i < spaces.size(); ++i )
    {
      m_pointerRecord->m_allocators[ internal::toChaiExecutionSpace( spaces.begin()[ i ] ) ] = allocators.begin()[ i ].getId();
    }
  #endif
  }

  /**
   * @brief Copy constructor.
   * @param src The buffer to copy.
   * @details In addition to performing a shallow copy of @p src if the chai execution space
   *   is set *this will contain a pointer the the allocation in that space.
   */
  LVARRAY_HOST_DEVICE inline
  ChaiBuffer( ChaiBuffer const & src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointerRecord( src.m_pointerRecord )
  {
  #if defined(LVARRAY_USE_DEVICE) && !defined(LVARRAY_DEVICE_COMPILE)
    move( internal::toMemorySpace( internal::getArrayManager().getExecutionSpace() ), true );
  #endif
  }

  /**
   * @copydoc ChaiBuffer( ChaiBuffer const & )
   * @param size The number of values in the allocation.
   * @note In addition to performing a shallow copy of @p src if the chai execution space
   *   is set *this will contain a pointer the the allocation in that space. It will also
   *   move any nested objects.
   */
  LVARRAY_HOST_DEVICE inline
  ChaiBuffer( ChaiBuffer const & src, std::ptrdiff_t const size ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointerRecord( src.m_pointerRecord )
  {
  #if defined(LVARRAY_USE_DEVICE) && !defined(LVARRAY_DEVICE_COMPILE)
    moveNested( internal::toMemorySpace( internal::getArrayManager().getExecutionSpace() ), size, true );
  #else
    LVARRAY_UNUSED_VARIABLE( size );
  #endif
  }

  /**
   * @brief Move constructor.
   * @param src The ChaiBuffer to be moved from, is uninitialized after this call.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  ChaiBuffer( ChaiBuffer && src ):
    m_pointer( src.m_pointer ),
    m_capacity( src.m_capacity ),
    m_pointerRecord( src.m_pointerRecord )
  {
    src.m_capacity = 0;
    src.m_pointer = nullptr;
    src.m_pointerRecord = nullptr;
  }

  /**
   * @brief Create a shallow copy of @p src but with a different type.
   * @tparam U The type to convert from.
   * @param src The buffer to copy.
   */
  template< typename U >
  LVARRAY_HOST_DEVICE inline constexpr
  ChaiBuffer( ChaiBuffer< U > const & src ):
    m_pointer( reinterpret_cast< T * >( src.data() ) ),
    m_capacity( typeManipulation::convertSize< T, U >( src.capacity() ) ),
    m_pointerRecord( &src.pointerRecord() )
  {}

  /**
   * @brief Copy assignment operator.
   * @param src The ChaiBuffer to be copied.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline LVARRAY_INTEL_CONSTEXPR
  ChaiBuffer & operator=( ChaiBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_pointer = src.m_pointer;
    m_pointerRecord = src.m_pointerRecord;
    return *this;
  }

  /**
   * @brief Move assignment operator.
   * @param src The ChaiBuffer to be moved from, is uninitialized after this call.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline LVARRAY_INTEL_CONSTEXPR
  ChaiBuffer & operator=( ChaiBuffer && src )
  {
    m_capacity = src.m_capacity;
    m_pointer = src.m_pointer;
    m_pointerRecord = src.m_pointerRecord;

    src.m_capacity = 0;
    src.m_pointer = nullptr;
    src.m_pointerRecord = nullptr;

    return *this;
  }

  /**
   * @brief Reallocate the buffer to the new capacity.
   * @param size the number of values that are initialized in the buffer. Values between [0, size) are destroyed.
   * @param space The space to perform the reallocation in. If space is the CPU then the buffer is reallocated
   *   only on the CPU and it is free'd in the other spaces. If the space is the GPU the the current size must be zero.
   * @param newCapacity the new capacity of the buffer.
   * @note Although it is marked as a host-device method, this is only valid to call from the host.
   */
  LVARRAY_HOST_DEVICE
  void reallocate( std::ptrdiff_t const size, MemorySpace const space, std::ptrdiff_t const newCapacity )
  {
  #if defined(LVARRAY_DEVICE_COMPILE)
    LVARRAY_ERROR( "Allocation from device is not supported." );
  #else
    move( space, true );
    chai::PointerRecord * const newRecord = new chai::PointerRecord{};
    newRecord->m_size = newCapacity * sizeof( T );
    newRecord->m_user_callback = m_pointerRecord->m_user_callback;

    for( int s = chai::CPU; s < chai::NUM_EXECUTION_SPACES; ++s )
    {
      newRecord->m_allocators[ s ] = m_pointerRecord->m_allocators[ s ];
    }

    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );

    internal::chaiLock.lock();
    internal::getArrayManager().allocate( newRecord, chaiSpace );
    internal::chaiLock.unlock();

    T * const newPointer = static_cast< T * >( newRecord->m_pointers[ chaiSpace ] );

    if( size > 0 )
    {
      LVARRAY_ERROR_IF_NE_MSG( space, MemorySpace::host, "Calling reallocate with a non-zero current size is not yet supporeted for the GPU." );
      std::ptrdiff_t const overlapAmount = std::min( newCapacity, size );
      arrayManipulation::uninitializedMove( newPointer, overlapAmount, m_pointer );
      arrayManipulation::destroy( m_pointer, size );
    }

    free();
    m_capacity = newCapacity;
    m_pointer = newPointer;
    m_pointerRecord = newRecord;
    registerTouch( space );
  #endif
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note To destroy the values and free the data call bufferManipulation::free.
   * @note Although it is marked as a host-device method, this is only valid to call from the host.
   */
  LVARRAY_HOST_DEVICE inline
  void free()
  {
  #if defined(LVARRAY_DEVICE_COMPILE)
    LVARRAY_ERROR( "Deallocation from device is not supported." );
  #else
    std::lock_guard< std::mutex > lock( internal::chaiLock );
    internal::getArrayManager().free( m_pointerRecord );
    m_capacity = 0;
    m_pointer = nullptr;
    m_pointerRecord = nullptr;
  #endif
  }

  /**
   * @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  std::ptrdiff_t capacity() const
  { return m_capacity; }

  /**
   * @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_pointer; }

  /**
   * @brief Return a reference to the associated CHAI PointerRecord.
   * @return A reference to the associated CHAI PointerRecord.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  chai::PointerRecord & pointerRecord() const
  { return *m_pointerRecord; }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @return The value at position @p i .
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
   * @note If they type T supports it this will call move( @p space, @p touch ) on each sub object.
   */
  inline
  void moveNested( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
  #if defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP )
    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );
    if( m_pointerRecord == nullptr ||
        m_capacity == 0 ||
        chaiSpace == chai::NONE ) return;

    chai::ExecutionSpace const prevSpace = m_pointerRecord->m_last_space;

    if( prevSpace == chai::CPU && prevSpace != chaiSpace ) moveInnerData( space, size, touch );

    move( space, touch );

    if( prevSpace == chai::GPU && prevSpace != chaiSpace ) moveInnerData( space, size, touch );
  #else
    LVARRAY_ERROR_IF_NE( space, MemorySpace::host );
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_UNUSED_VARIABLE( touch );
  #endif
  }

  /**
   * @brief Move the buffer to the given execution space, optionally touching it.
   * @param space The space to move the buffer to.
   * @param touch If the buffer should be touched in the new space or not.
   * @note This will not move subobjects.
   */
  void move( MemorySpace const space, bool const touch ) const
  {
  #if defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP)
    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );
    if( m_pointerRecord == nullptr ||
        m_capacity == 0 ||
        chaiSpace == chai::NONE ) return;

    auto & am = internal::getArrayManager();
    const_cast< T * & >( m_pointer ) =
      static_cast< T * >( am.move( const_cast< T_non_const * >( m_pointer ), m_pointerRecord, chaiSpace ) );

    if( !std::is_const< T >::value && touch ) m_pointerRecord->m_touched[ chaiSpace ] = true;
    m_pointerRecord->m_last_space = chaiSpace;
  #else
    LVARRAY_ERROR_IF_NE( space, MemorySpace::host );
    LVARRAY_UNUSED_VARIABLE( touch );
  #endif
  }

  /**
   * @return The last space the ChaiBuffer was moved to.
   */
  MemorySpace getPreviousSpace() const
  { return internal::toMemorySpace( m_pointerRecord->m_last_space ); }

  /**
   * @brief Touch the buffer in the given space.
   * @param space the space to touch.
   */
  inline constexpr
  void registerTouch( MemorySpace const space ) const
  {
    chai::ExecutionSpace const chaiSpace = internal::toChaiExecutionSpace( space );
    m_pointerRecord->m_touched[ chaiSpace ] = true;
    m_pointerRecord->m_last_space = chaiSpace;
  }

  /**
   * @tparam U The type of the owning class, will be displayed in the callback.
   * @brief Set the name associated with this buffer which is used in the chai callback.
   * @param name the of the buffer.
   */
  template< typename U=ChaiBuffer< T > >
  void setName( std::string const & name )
  {
    std::string const typeString = system::demangleType< U >();
    m_pointerRecord->m_user_callback =
      [name, typeString]( chai::PointerRecord const * const record, chai::Action const act, chai::ExecutionSpace const s )
    {
      if( act == chai::ACTION_MOVE )
      {
        std::string const size = system::calculateSize( record->m_size );
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
   * @return void.
   * @note This method is only active when T has a method move( MemorySpace ).
   */
  template< typename U=T_non_const >
  std::enable_if_t< bufferManipulation::HasMemberFunction_move< U > >
  moveInnerData( MemorySpace const space, std::ptrdiff_t const size, bool const touch ) const
  {
    if( space == MemorySpace::undefined ) return;

    for( std::ptrdiff_t i = 0; i < size; ++i )
    {
      const_cast< T_non_const * >( m_pointer )[ i ].move( space, touch );
    }
  }

  /**
   * @tparam U A dummy parameter to enable SFINAE, do not specify.
   * @brief Move inner allocations to the memory space @p space.
   * @return void.
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
  chai::PointerRecord * m_pointerRecord = nullptr;
};

} /* namespace LvArray */
