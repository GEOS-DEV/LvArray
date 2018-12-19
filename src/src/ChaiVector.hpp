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

#include "CXX_UtilsConfig.hpp"

#include <type_traits>
#include <iterator>

#ifdef USE_CHAI
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"
#include <mutex>
#else
#include <cstdlib>
#endif


#ifdef USE_CHAI
namespace internal
{
static std::mutex chai_lock;
}
#endif

template < typename T >
class ChaiVector
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  using size_type = size_t;
  using iterator = T *;
  using const_iterator = T const *;

  /**
   * @brief Default constructor, creates a new empty vector.
   */
  ChaiVector():
#ifdef USE_CHAI
    m_array(),
#else
    m_array( nullptr ),
    m_capacity( 0 ),
#endif
    m_length( 0 )
  {
    registerTouch(chai::CPU);
  }

  /**
   * @brief Creates a new vector of the given length.
   * @param [in] initial_length the initial length of the vector.
   */
  ChaiVector( size_type initial_length ):
#ifdef USE_CHAI
    m_array(),
#else
    m_array( nullptr ),
    m_capacity( 0 ),
#endif
    m_length( 0 )
  {
    resize( initial_length );
    registerTouch(chai::CPU);
  }

  /**
   * @brief Copy constructor, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   * @note The copy is a shallow copy and newly constructed ChaiVector doesn't own the data,
   * as such using push_back or other methods that change the state of the array is dangerous.
   * @note When using multiple memory spaces using the copy constructor can trigger a move.
   */
  LVARRAY_HOST_DEVICE ChaiVector( const ChaiVector& source ):
    m_array( source.m_array ),
#ifndef USE_CHAI
    m_capacity( source.capacity() ),
#endif
    m_length( source.m_length )
  {}

  /**
   * @brief Move constructor, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @note Unlike the copy constructor this can not trigger a memory movement.
   */
  ChaiVector( ChaiVector&& source ):
    m_array( source.m_array ),
#ifndef USE_CHAI
    m_capacity( source.capacity() ),
#endif
    m_length( source.m_length )
  {
#ifndef USE_CHAI
    source.m_capacity = 0;
#endif
    source.m_array = nullptr;
    source.m_length = 0;
  }

  /**
   * @brief Free's the data.
   */
  void free()
  {
    if( capacity() > 0 )
    {
      clear();
#ifdef USE_CHAI
      internal::chai_lock.lock();
      m_array.free();
      internal::chai_lock.unlock();
#else
      std::free( m_array );
      m_capacity = 0;
#endif
    }

    m_array = nullptr;
    m_length = 0;
  }

  /**
   * @brief Let go of any memory held. Should not be called lightly, does not free
   *  said memory.
   */
  void reset()
  {
    m_array = nullptr;
    m_length = 0;
#ifndef USE_CHAI
    m_capacity = 0;
#endif
  }

  /**
   * @brief Copy assignment operator, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE ChaiVector& operator=( ChaiVector const& source )
  {
    m_array = source.m_array;
    m_length = source.size();
#ifndef USE_CHAI
    m_capacity = source.capacity();
#endif
    return *this;
  }

  /**
   * @brief Dereference operator for the underlying active pointer.
   * @param [in] pos the index to access.
   * @return a reference to the value at the given index.
   */
  LVARRAY_HOST_DEVICE T & operator[]( size_type pos ) const
  { return m_array[ pos ]; }

  /**
   * @brief Return a reference to the first value in the array.
   */
  T& front() const
  { return m_array[0]; }

  /**
   * @brief Return a reference to the last value in the array.
   */
  T& back() const
  { return m_array[ size() - 1 ]; }

  /**
   * @brief Return a pointer to the data.
   */
  LVARRAY_HOST_DEVICE T* data() const
  { return &m_array[0]; }

  /**
   * @brief Return a random access iterator to the beginning of the vector.
   */
  iterator begin() const
  { return &front(); }

  /**
   * @brief Return a random access iterator to one past the end of the vector.
   */
  iterator end() const
  { return &back() + 1; }

  /**
   * @brief Return true iff the vector holds no data.
   */
  bool empty() const
  { return size() == 0; }

  /**
   * @brief Return the number of values held in the vector.
   */
  LVARRAY_HOST_DEVICE size_type size() const
  { return m_length; }

  /**
   * @brief Allocate space to hold at least the given number of values.
   * @param [in] new_cap the new capacity.
   */
  void reserve( size_type new_cap )
  {
    if( new_cap > capacity() )
    {
      realloc( new_cap );
    }
  }

  /**
   * @brief Return the capacity of the vector.
   */
  size_type capacity() const
  {
#ifdef USE_CHAI
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
    return insert( index, value );
  }

  /**
   * @brief Insert the given value at the given position.
   * @param [in] pos the position at which to insert the value.
   * @param [in] value the value to insert.
   * @return An iterator to the position at which the insertion was done.
   */
  iterator insert( const_iterator pos, const T&& value )
  {
    const size_type index = pos - begin();
    return insert( index, std::move( value ) );
  }

  /**
   * @brief Insert the given value at the given position.
   * @param [in] pos the position at which to insert the value.
   * @param [in] value the value to insert.
   * @return An iterator to the position at which the insertion was done.
   */
  iterator insert( size_type index, const T& value )
  {
    emplace( index, 1 );
    new ( &m_array[ index ] ) T( value );
    return begin() + index;
  }

  /**
   * @brief Insert the given value at the given position.
   * @param [in] pos the position at which to insert the value.
   * @param [in] value the value to insert.
   * @return An iterator to the position at which the insertion was done.
   */
  iterator insert( size_type index, const T&& value )
  {
    emplace( index, 1 );
    new ( &m_array[ index ] ) T( std::move( value ));
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
    size_type index = pos - begin();
    const size_type n = std::distance( first, last );
    emplace( index, n );

    /* Initialize the newly vacant values moved out of to the default value. */
    for( ; first != last ; ++first )
    {
      new ( &m_array[ index++ ] ) T( *first );
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
    for( size_type i = index ; i < size() - 1 ; ++i )
    {
      m_array[ i ] = std::move( m_array[ i + 1 ] );
    }

    m_array[ size() - 1 ].~T();
    --m_length;

    return begin() + index;
  }

  /**
   * @brief Append a value to the end of the array.
   * @param [in] value the value to append.
   */
  void push_back( T const & value )
  {
    if( size() + 1 > capacity() )
    {
      dynamicRealloc( size() + 1 );
    }

    new ( &m_array[ size() ] ) T( value );
    ++m_length;
  }

  /**
   * @brief Append a value to the end of the array.
   * @param [in] value the value to append.
   */
  void push_back( T && value )
  {
    if( size() + 1 > capacity() )
    {
      dynamicRealloc( size() + 1 );
    }

    new ( &m_array[ size() ] ) T( std::move( value ));
    ++m_length;
  }

  /**
   * @brief Delete the last value.
   */
  void pop_back()
  {
    if( size() > 0 )
    {
      --m_length;
      m_array[ size() ].~T();
    }
  }

  /**
   * @brief Resize the vector to the new length.
   * @param [in] new_length the new length of the vector.
   * @note If reducing the size the values past the new size are destroyed,
   * if increasing the size the values past the current size are initialized with
   * the default constructor.
   */
  void resize( const size_type new_length, T const & defaultValue  = T())
  {
    if( new_length > capacity() )
    {
      realloc( new_length );
    }

    /* Delete things between new_length and size() */
    for( size_type i = new_length ; i < size() ; ++i )
    {
      m_array[ i ].~T();
    }

    /* Initialize things size() and new_length */
    for( size_type i = size() ; i < new_length ; ++i )
    {
      new ( &m_array[ i ] ) T( defaultValue );
    }

    m_length = new_length;
  }

#ifdef USE_CHAI
  void move( chai::ExecutionSpace space )
  {
    m_array.move( space );
    registerTouch( space );
  }
#endif

private:

  void registerTouch( chai::ExecutionSpace space )
  {
#ifdef USE_CHAI
    m_array.registerTouch( space );
#else
    ((void) space);
#endif
  }

  /**
   * @brief Insert the given number of default values at the given position.
   * @param [in] n the number of values to insert.
   * @param [in] pos the position at which to insert.
   */
  void emplace( size_type pos, size_type n )
  {
    if( n == 0 )
    {
      return;
    }

    size_type new_length = size() + n;
    if( new_length > capacity() )
    {
      dynamicRealloc( new_length );
    }

    /* Move the existing values down by n. */
    for( size_type i = size() ; i > pos ; --i )
    {
      const size_type cur_pos = i - 1;
      new ( &m_array[ cur_pos + n ] ) T( std::move( m_array[ cur_pos ] ) );
    }

    m_length = new_length;
  }

  /**
   * @brief Reallocate the underlying array to have the given capacity.
   * @param [in] new_capacity the new capacity.
   */
  void realloc( size_type new_capacity )
  {
#ifdef USE_CHAI
    internal::chai_lock.lock();
    chai::ManagedArray<T> new_array( new_capacity );
    internal::chai_lock.unlock();
#else
    T* new_array = static_cast< T* >( std::malloc( new_capacity * sizeof( T ) ) );
#endif

    const size_type new_size = new_capacity > size() ? size() : new_capacity;
    for( size_type i = 0 ; i < new_size ; ++i )
    {
      new ( &new_array[ i ] ) T( std::move( m_array[ i ] ) );
    }

    for( size_type i = 0 ; i < size() ; ++i )
    {
      m_array[ i ].~T();
    }

#ifdef USE_CHAI
    if( capacity() != 0 )
    {
      internal::chai_lock.lock();
      m_array.free();
      internal::chai_lock.unlock();
    }
#else
    std::free( m_array );
    m_capacity = new_capacity;
#endif
    m_array = new_array;
    registerTouch(chai::CPU);
  }

  /**
   * @brief Performs a dynamic reallocation, which makes the capacity twice the new length.
   * @param [in] new_length the new length.
   */
  void dynamicRealloc( size_type new_length )
  { reserve( 2 * new_length ); }

#ifdef USE_CHAI
  chai::ManagedArray<T> m_array;
#else
  T* m_array;
  size_type m_capacity;
#endif
  size_type m_length;
};

#endif /* CHAI_VECTOR_HPP_ */
