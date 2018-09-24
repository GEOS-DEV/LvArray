/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#ifndef CHAI_VECTOR_HPP_
#define CHAI_VECTOR_HPP_

#include <type_traits>
#include <iterator>

#ifdef GEOSX_USE_CHAI
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"
#include <mutex>
#else
#include <cstdlib>
#endif


#ifdef GEOSX_USE_CHAI
namespace internal
{
static std::mutex chai_lock;
}
#endif

template < typename T >
class ChaiVector 
#ifdef GEOSX_USE_CHAI
: public chai::CHAICopyable
#endif
{
public:

  using value_type = T;
  using size_type = size_t;
  using reference = T&;
  using const_reference = const T&;
  using rvalue_reference = T&&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = pointer;
  using const_iterator = const_pointer;


  /**
   * @brief Default constructor, creates a new empty vector.
   */
  ChaiVector() :
#ifdef GEOSX_USE_CHAI
    m_array(),
#else
    m_array( nullptr ),
    m_capacity( 0 ),
#endif
    m_length( 0 ),
    m_copied( false )
  {}

  /**
   * @brief Creates a new vector of the given length.
   * @param [in] initial_length the initial length of the vector.
   */
  ChaiVector( size_type initial_length ) :
#ifdef GEOSX_USE_CHAI
    m_array(),
#else
    m_array( nullptr ),
    m_capacity( 0 ),
#endif
    m_length( 0 ),
    m_copied( false )
  {
    resize( initial_length );
  }

  /**
   * @brief Copy constructor, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   * @note The copy is a shallow copy and newly constructed ChaiVector doesn't own the data,
   * as such using push_back or other methods that change the state of the array is dangerous.
   * @note When using multiple memory spaces using the copy constructor can trigger a move.
   */
  ChaiVector( const ChaiVector& source ) :
    m_array( source.m_array ),
#ifndef GEOSX_USE_CHAI
    m_capacity( source.capacity() ),
#endif
    m_length( source.m_length ),
    m_copied( true )
  {}

  /**
   * @brief Move constructor, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @note Unlike the copy constructor this can not trigger a memory movement.
   */
  ChaiVector( ChaiVector&& source ) :
    m_array( source.m_array ),
#ifndef GEOSX_USE_CHAI
    m_capacity( source.capacity() ),
#endif
    m_length( source.m_length ),
    m_copied( source.m_copied )
  {
#ifndef GEOSX_USE_CHAI
    source.m_capacity = 0;
#endif
    source.m_array = nullptr;
    source.m_length = 0;
    source.m_copied = true;
  }

  /**
   * @brief Destructor, will destroy the objects and free the memory if it owns the data.
   */
  ~ChaiVector()
  {
    if ( capacity() > 0 && !m_copied )
    {
      clear();
#ifdef GEOSX_USE_CHAI
      internal::chai_lock.lock();
      m_array.free();
      internal::chai_lock.unlock();
#else
      std::free( m_array );
#endif
    }
  }

  /**
   * @brief Move assignment operator, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @return *this.
   */
  ChaiVector& operator=( ChaiVector const& source )
  {
    if ( m_copied )
    {
#ifdef GEOSX_USE_CHAI
      m_array = chai::ManagedArray<T>();
#else
      m_array = nullptr;
      m_capacity = 0;
#endif
    }

    m_copied = false;
    resize( source.size() );

    for ( size_type i = 0; i < size(); ++i )
    {
      m_array[ i ] = source[ i ];
    }

    return *this;
  }

  /**
   * @brief Move assignment operator, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @return *this.
   */
  ChaiVector& operator=( ChaiVector&& source )
  {
    if ( capacity() > 0 && !m_copied )
    {
      clear();
#ifdef GEOSX_USE_CHAI
      internal::chai_lock.lock();
      m_array.free();
      internal::chai_lock.unlock();
#else
      std::free( m_array );
#endif
    }

    m_array = source.m_array;
    m_length = source.m_length;
    m_copied = source.m_copied;

#ifndef GEOSX_USE_CHAI
    m_capacity = source.m_capacity;
    source.m_capacity = 0;
#endif

    source.m_array = nullptr;
    source.m_length = 0;
    source.m_copied = true;
    return *this;
  }

  /**
   * @brief Return if this ChaiVector is a copy and therefore does not own its data.
   */
  bool isCopy() const
  { return m_copied; }

  /**
   * @brief Dereference operator for the underlying active pointer.
   * @param [in] pos the index to access.
   * @return a reference to the value at the given index.
   */
  /// @{
  reference operator[]( size_type pos )
  { return m_array[ pos ]; }

  const_reference operator[]( size_type pos ) const
  { return m_array[ pos ]; }
  /// @}

  /**
   * @brief Return a reference to the first value in the array.
   */
  /// @{
  reference front()
  { return m_array[0]; }

  const_reference front() const
  { return m_array[0]; }
  /// @}

  /**
   * @brief Return a reference to the last value in the array.
   */
  /// @{
  reference back()
  { return m_array[ m_length - 1 ]; }

  const_reference back() const
  { return m_array[ m_length  - 1 ]; }
  /// @}

  /**
   * @brief Return a pointer to the data.
   */
  /// @{
  pointer data()
  { return &m_array[0]; }

  const_pointer data() const
  { return &m_array[0]; }
  /// @}

  /**
   * @brief Return a random access iterator to the beginning of the vector.
   */
  /// @{
  iterator begin()
  { return &front(); }

  const_iterator begin() const
  { return &front(); }
  /// @}

  /**
   * @brief Return a random access iterator to one past the end of the vector.
   */
  /// @{
  iterator end()
  { return &back() + 1; }

  const_iterator end() const
  { return &back() + 1; }
  /// @}

  /**
   * @brief Return true iff the vector holds no data.
   */
  bool empty() const
  { return size() == 0; }

  /**
   * @brief Return the number of values held in the vector.
   */
  size_type size() const
  { return m_length; }

  /**
   * @brief Allocate space to hold at least the given number of values.
   * @param [in] new_cap the new capacity.
   */
  void reserve( size_type new_cap )
  {
    if ( new_cap > capacity() )
    {
      realloc( new_cap );
    }
  }

  /**
   * @brief Return the capacity of the vector.
   */
  size_type capacity() const
  { 
#ifdef GEOSX_USE_CHAI
    return m_array.size();
#else
    return m_capacity;
#endif  
  }

  /**
   * @brief Resize the vector to length zero.
   * @note Does not change the capacity.
   */
  void clear()
  { resize( 0 ); }

  /**
   * @brief Insert the given value at the given position.
   * @param [in] pos the position at which to insert the value.
   * @param [in] value the value to insert.
   * @return An iterator to the position at which the insertion was done.
   */
  iterator insert( const_iterator pos, const T& value )
  {
    const size_type index = pos - begin();
    emplace( 1, index );
    m_array[ index ] = value;
    return begin() + index;
  }

  /**
   * @brief Insert the given values starting at the give position.
   * @param [in] pos the position at which to begin the insertion.
   * @param [in] first iterator to the first value to insert.
   * @param [in] last iterator to one past the last value to insert.
   * @return An iterator to the position at which the insertion was done.
   */
  template < typename InputIt >
  iterator insert( const_iterator pos, InputIt first, InputIt last )
  {
    const size_type index = pos - begin();
    const size_type n = std::distance( first, last );
    emplace( n, index );

    for( size_type i = 0; i < n; ++i )
    {
      m_array[ index + i ] = *first;
      first++;
    }

    return begin() + index;
  }

  /**
   * @brief Remove the value at the given position and shift down all subsequent values.
   * @param [in] pos the position to remove.
   * @return An iterator to the position at which the erase was done.
   */
  iterator erase( const_iterator pos )
  {
    const size_type index = pos - begin();
    m_length--;
    for ( size_type i = index; i < m_length; ++i )
    {
      m_array[ i ] = std::move( m_array[ i + 1 ] );
    }

    m_array[ m_length ].~T();

    return begin() + index;
  }

  /**
   * @brief Append a value to the end of the array.
   * @param [in] val the value to append.
   */
  void push_back( const_reference value )
  {
    m_length++;
    if ( m_length > capacity() )
    {
      dynamicRealloc( m_length );
    }

    new ( &m_array[ m_length - 1 ] ) T();
    m_array[ m_length - 1 ] = value;
  }

  /**
   * @brief Delete the last value.
   */
  void pop_back()
  {
    if ( m_length > 0 )
    {
      m_length--;
      m_array[ m_length ].~T();
    }
  }

  /**
   * @brief Resize the vector to the new length.
   * @param [in] new_length the new length of the vector.
   * @note If reducing the size the values past the new size are destroyed,
   * if increasing the size the values past the current size are initialized with
   * the default constructor.
   */
  void resize( const size_type new_length )
  {
    if ( new_length > capacity() )
    {
      realloc( new_length );
    }

    if ( new_length < m_length )
    {
      for ( size_type i = new_length; i < m_length; ++i )
      {
        m_array[ i ].~T();
      }
    }
    else
    {
      for ( size_type i = m_length; i < new_length; ++i )
      {
        new ( &m_array[ i ] ) T();
      }
    }

    m_length = new_length;
  }

private:

  /**
   * @brief Insert the given number of default values at the given position.
   * @param [in] n the number of values to insert.
   * @param [in] pos the position at which to insert.
   */
  void emplace( size_type n, size_type pos )
  {
    if ( n == 0 )
    {
      return;
    }

    size_type new_length = m_length + n;
    if ( new_length > capacity() )
    {
      dynamicRealloc( new_length );
    }

    /* Move the existing values down by n. */
    for ( size_type i = m_length; i > pos; --i )
    {
      const size_type cur_pos = i - 1;
      new ( &m_array[ cur_pos + n ] ) T( std::move( m_array[ cur_pos ] ) );
    }

    /* Initialize the newly vacant values moved out of to the default value. */
    for ( size_type i = 0; i < n; ++i )
    {
      const size_type cur_pos = pos + i;
      new ( &m_array[ cur_pos ] ) T();
    }

    m_length = new_length;
  }

  /**
   * @brief Reallocate the underlying array to have the given capacity.
   * @param [in] new_capacity the new capacity.
   */
  void realloc( size_type new_capacity )
  {
#ifdef GEOSX_USE_CHAI
    internal::chai_lock.lock();
    const size_type initial_capacity = capacity();
    if ( capacity() == 0 )
    {
      m_array.allocate( new_capacity );
    }
    else
    {
      m_array.reallocate( new_capacity );
    }
    internal::chai_lock.unlock();
#else
    m_array = static_cast< T* >( std::realloc( static_cast< void* >( m_array ), new_capacity * sizeof( T ) ) );
    m_capacity = new_capacity;
#endif
  }

  /**
   * @brief Performs a dynamic reallocation, which makes the capacity twice the new length.
   * @param [in] new_length the new length.
   */
  void dynamicRealloc( size_type new_length )
  { reserve( 2 * new_length ); }

#ifdef GEOSX_USE_CHAI
  chai::ManagedArray<T> m_array;
#else
  T* m_array;
  size_type m_capacity;
#endif
  size_type m_length;
  bool m_copied;
};

#endif /* CHAI_VECTOR_HPP_ */
