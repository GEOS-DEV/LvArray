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

#ifndef CHAI_VECTOR_HPP_
#define CHAI_VECTOR_HPP_

#include "CXX_UtilsConfig.hpp"
#include "Logger.hpp"
#include "arrayManipulation.hpp"

#include <type_traits>
#include <iterator>

#ifdef USE_CHAI
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"
#include <mutex>
#else
#include <cstdlib>
#endif

namespace LvArray
{

template< class COL_TYPE, class INDEX_TYPE >
class SparsityPattern;

#ifdef USE_CHAI
namespace internal
{
static ::std::mutex chai_lock;
}
#endif

template< typename T >
class ChaiVector
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:

  template< class U, class INDEX_TYPE >
  friend class SortedArray;

  template< class COL_TYPE, class INDEX_TYPE >
  friend class SparsityPattern;

  template< class U, class COL_TYPE, class INDEX_TYPE >
  friend class CRSMatrix;

  template< class U, class INDEX_TYPE, bool CONST_SIZES >
  friend class ArrayOfArraysView;

  template< class U, class INDEX_TYPE >
  friend class ArrayOfArrays;

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
  {}

  /**
   * @brief Creates a new vector of the given length.
   * @param [in] initialLength the initial length of the vector.
   */
  ChaiVector( size_type initialLength ):
#ifdef USE_CHAI
    m_array(),
#else
    m_array( nullptr ),
    m_capacity( 0 ),
#endif
    m_length( 0 )
  {
    resize( initialLength );
  }

  /**
   * @brief Use this to create an empty ChaiVector.
   * @note You cannot call any methdods on the ChaiVector until it has been
   *       initialized from another ChaiVector using either of the operator= methods.
   */
  ChaiVector( std::nullptr_t ):
    m_array( nullptr ),
  #ifndef USE_CHAI
    m_capacity( 0 ),
  #endif
    m_length( 0 )
  {}

  /**
   * @brief Copy constructor, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   *
   * @note The copy is a shallow copy and newly constructed ChaiVector doesn't own the data,
   *        as such using push_back or other methods that change the state of the array is dangerous.
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
   *
   * @note Unlike the copy constructor this can not trigger a memory movement.
   */
  LVARRAY_HOST_DEVICE ChaiVector( ChaiVector&& source ):
    m_array( nullptr ),
#ifndef USE_CHAI
    m_capacity( source.capacity() ),
#endif
    m_length( source.m_length )
  {
    m_array = source.m_array;
#ifndef USE_CHAI
    source.m_capacity = 0;
#endif
    source.m_array = nullptr;
    source.m_length = 0;
  }

  template< class U=T >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if< !std::is_const< U >::value,
                                    ChaiVector< T const > const & >::type
    () const restrict_this
  { return reinterpret_cast< ChaiVector< T const > const & >(*this); }

  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ChaiVector< T const > const & toConst() const restrict_this
  { return *this; }

  /**
   * @brief Perform a deep copy front this in to dst.
   * @param [in] dst the ChaiVector to copy into.
   */
  template< class U >
  void copy_into( ChaiVector< U >& dst ) const
  {
    dst.resize( size() );
    for( size_type i = 0 ; i < size() ; ++i )
    {
      dst[ i ] = m_array[ i ];
    }
  }

  /**
   * @brief Free's the data.
   */
  void free()
  {
    clear();
    releaseAllocation();

    m_array = nullptr;
    m_length = 0;
  }

  /**
   * @brief Let go of any memory held. Should not be called lightly, does not free
   *        said memory.
   */
  LVARRAY_HOST_DEVICE void reset()
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
   * @brief Copy assignment operator, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   */
  LVARRAY_HOST_DEVICE ChaiVector& operator=( ChaiVector && source )
  {
    m_array = std::move( source.m_array );
    source.m_array = nullptr;

    m_length = source.m_length;
    source.m_length = 0;

#ifndef USE_CHAI
    m_capacity = source.m_capacity;
    source.m_capacity = 0;
#endif

    return *this;
  }

  /**
   * @tparam INDEX_TYPE the integer used to index into the array.
   * @brief Dereference operator for the underlying active pointer.
   * @param [in] pos the index to access.
   * @return a reference to the value at the given index.
   */
  template< class INDEX_TYPE >
  LVARRAY_HOST_DEVICE T & operator[]( INDEX_TYPE pos ) const
  { return m_array[ pos ]; }

  /**
   * @brief Return a pointer to the data.
   */
  LVARRAY_HOST_DEVICE T* data() const
  { return &m_array[0]; }

  /**
   * @brief Return true iff the vector holds no data.
   */
  LVARRAY_HOST_DEVICE
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
  LVARRAY_HOST_DEVICE size_type capacity() const
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
   */
  void insert( size_type index, const T& value )
  {
    dynamicResize( size() + 1 );

    arrayManipulation::insert( data(), m_length, index, value );
    m_length += 1;
  }

  /**
   * @brief Insert the given value at the given position.
   * @param [in] pos the position at which to insert the value.
   * @param [in] value the value to insert.
   */
  void insert( size_type index, const T&& value )
  {
    dynamicResize( size() + 1 );

    arrayManipulation::insert( data(), m_length, index, std::move( value ));
    m_length += 1;
  }

  /**
   * @brief Insert the given values starting at the give position.
   * @param [in] pos the position at which to begin the insertion.
   * @param [in] first iterator to the first value to insert.
   * @param [in] last iterator to one past the last value to insert.
   */
  void insert( size_type index, T const * const values, size_type n )
  {
    dynamicResize( size() + n );

    arrayManipulation::insert( data(), m_length, index, values, n );
    m_length += n;
  }

  /**
   * @brief Append a value to the end of the array.
   * @param [in] value the value to append.
   */
  void push_back( T const & value )
  {
    dynamicResize( size() + 1 );

    arrayManipulation::append( data(), m_length, value );
    m_length += 1;
  }

  /**
   * @brief Append a value to the end of the array.
   * @param [in] value the value to append.
   */
  void push_back( T && value )
  {
    dynamicResize( size() + 1 );

    arrayManipulation::append( data(), m_length, std::move( value ));
    m_length += 1;
  }

  /**
   * @brief Append multiple values to the end of the array.
   * @param [in] values the values to append.
   * @param [in] n_values the number of values to append.
   */
  void push_back( T const * values, size_type n_values )
  {
    dynamicResize( size() + n_values );

    arrayManipulation::append( data(), m_length, values, n_values );
    m_length += n_values;
  }

  /**
   * @brief Delete the last value.
   */
  void pop_back()
  {
    arrayManipulation::popBack( data(), m_length );
    m_length -= 1;
  }

  /**
   * @brief Resize the vector to the new length.
   * @param [in] newLength the new length of the vector.
   *
   * @note If reducing the size the values past the new size are destroyed,
   *       if increasing the size the values past the current size are initialized with
   *       the default constructor.
   */
  template< class ... ARGS >
  void resize( const size_type newLength, ARGS && ... args )
  {
    if( newLength > capacity() )
    {
      realloc( newLength );
    }

    arrayManipulation::resize( data(), m_length, newLength, std::forward< ARGS >( args )... );
    m_length = newLength;

    if( m_length > 0 )
    {
#ifdef USE_CHAI
      registerTouch( chai::CPU );
#endif
    }
  }

#ifdef USE_CHAI
  /**
   * @brief Move the array to the given memory space.
   * @param [in] space the space to move to.
   */
  void move( chai::ExecutionSpace space )
  {
    m_array.move( space );
    registerTouch( space );
  }
#endif

  /**
   * @brief Insert the given number of default values at the given position.
   * @param [in] pos the position at which to insert.
   * @param [in] n the number of values to insert.
   */
  void emplace( size_type pos, size_type n, T const & defaultValue = T() )
  {
    if( n == 0 ) return;

    dynamicResize( size() + n );

    arrayManipulation::emplace( data(), m_length, pos, n, defaultValue );
    m_length += n;
  }

  /**
   * @brief Remove the given number of values starting at the given position.
   * @param [in] pos the position at which to remove.
   * @param [in] n the number of values to remove.
   */
  void erase( size_type pos, size_type n=1 )
  {
    arrayManipulation::erase( data(), m_length, pos, n );
    m_length -= n;
  }

  void fill( T const & value )
  {
    for( size_type i = 0 ; i < size() ; ++i )
    {
      m_array[i] = value;
    }
  }

private:

  /**
   * @brief Free the allocated memory without destroying the values.
   *
   * @note This is used by the CRSMatrix after it destroys the values manually.
   */
  void releaseAllocation()
  {
#ifdef USE_CHAI
    internal::chai_lock.lock();
    m_array.free();
    internal::chai_lock.unlock();
#else
    std::free( m_array );
    m_capacity = 0;
#endif

    m_array = nullptr;
    m_length = 0;
  }

  /**
   * @brief Set the size of the array without doing any of the other machinery.
   *
   * @note This is used by the CRSMatrix and SortedArray.
   */
  void setSize( size_type newLength )
  {
    GEOS_ASSERT( arrayManipulation::isPositive( newLength ) && newLength <= capacity());
    m_length = newLength;
  }

  /**
   * @brief Insert n uninitialized values at the given position.
   * @param [in] pos the position at which to insert.
   * @param [in] n the number of values to insert.
   *
   * @note The newly inserted values are uninitialized hence placement new must be used.
   * @note This is used by the CRSMatrix and SparsityPattern classes.
   */
  void shiftUp( size_type pos, size_type n )
  {
    if( n == 0 ) return;

    dynamicResize( size() + n );

    arrayManipulation::shiftUp( data(), m_length, pos, n );
    m_length += n;
  }

  /**
   * @brief Shift the values in the array at or above the given position down by the given amount overwriting
   *        the existing values. The n entries at the end of the array are not destroyed.
   * @param [in] pos the ps at which to begin the shift.
   * @param [in] n the number of places to shift.
   *
   * @note The newly values at the end of the array are not destroyed.
   * @note This is used by the CRSMatrix class.
   */
  void shiftDown( size_type pos, size_type n )
  {
    if( n == 0 ) return;

    arrayManipulation::shiftDown( data(), m_length, pos, n );
    m_length -= n;
  }

#ifdef USE_CHAI
  /**
   * @brief Register a touch of the chai::ManagedArray in the given space.
   * @param [in] space the space to touch the array in.
   */
  void registerTouch( chai::ExecutionSpace space )
  {
    m_array.registerTouch( space );
  }
#endif

  /**
   * @brief Reallocate the underlying array to have the given capacity.
   * @param [in] new_capacity the new capacity.
   */
  void realloc( size_type new_capacity )
  {
#ifdef USE_CHAI
    internal::chai_lock.lock();
    chai::ManagedArray< T > new_array( new_capacity );
    internal::chai_lock.unlock();
#else
    T* new_array = static_cast< T* >( std::malloc( new_capacity * sizeof( T ) ) );
#endif

    /* Move the values over into the new array. */
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
    internal::chai_lock.lock();
    m_array.free();
    internal::chai_lock.unlock();
#else
    std::free( m_array );
    m_capacity = new_capacity;
#endif
    m_array = new_array;

#ifdef USE_CHAI
    registerTouch( chai::CPU );
#endif
  }

  /**
   * @brief Performs a dynamic reallocation if the new length will exceed the capacity.
   * @param [in] newLength the new length.
   * @note this doesn't actually set m_length.
   */
  void dynamicResize( size_type newLength )
  {
    if( newLength > capacity() )
    {
      dynamicRealloc( newLength );
    }
  }

  /**
   * @brief Performs a dynamic reallocation, which makes the capacity twice the new length.
   * @param [in] newLength the new length.
   */
  void dynamicRealloc( size_type newLength )
  { reserve( 2 * newLength ); }

#ifdef USE_CHAI
  chai::ManagedArray< T > m_array;
#else
  T* m_array;
  size_type m_capacity;
#endif
  size_type m_length;
};

} /* namespace LvArray */

#endif /* CHAI_VECTOR_HPP_ */
