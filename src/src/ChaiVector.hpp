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

#include <type_traits>
#include <iterator>
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"


template < typename T >
class ChaiVector
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


  /* Constructors. */
  ChaiVector() :
    m_array(),
    m_length( 0 ),
    m_copied( false )
  {}

  ChaiVector( size_type initial_length ) :
    m_array( initial_length ),
    m_length( 0 ),
    m_copied( false )
  {
    resize( initial_length );
  }

  ChaiVector( const ChaiVector& source ) :
    m_array( source.m_array ),
    m_length( source.m_length ),
    m_copied( true )
  {}

  ChaiVector( ChaiVector&& source ) :
    m_array( std::move( source.m_array ) ),
    m_length( source.m_length ),
    m_copied( source.m_copied )
  {
    source.m_length = 0;
  }

  ~ChaiVector()
  {
    if ( capacity() > 0 && !m_copied )
    {
      clear();
      m_array.free();
    }
  }

  ChaiVector& operator=( ChaiVector&& source )
  {
    m_array = std::move( source.m_array );
    m_length = source.m_length;
    m_copied = source.m_copied;
    source.m_length = 0;
    return *this;
  }

  /* Element access. */

  reference operator[]( size_type pos )
  { return m_array[ pos ]; }

  const_reference operator[]( size_type pos ) const
  { return m_array[ pos ]; }

  reference front()
  { return m_array[0]; }

  const_reference front() const
  { return m_array[0]; }

  reference back()
  { return m_array[ m_length - 1 ]; }

  const_reference back() const
  { return m_array[ m_length  - 1 ]; }

  pointer data()
  { return &m_array[0]; }

  const_pointer data() const
  { return &m_array[0]; }


  /* Iterators. */


  iterator begin()
  { return &m_array[0]; }

  const_iterator begin() const
  { return &m_array[0]; }

  iterator end()
  { return &m_array[ m_length ]; }

  const_iterator end() const
  { return &m_array[ m_length ]; }


  /* Capacity */


  bool empty() const
  { return size() == 0; }

  size_type size() const
  { return m_length; }

  void reserve( size_type new_cap )
  {
    if ( new_cap > capacity() )
    {
      realloc( new_cap );
    }
  }

  size_type capacity() const
  { return m_array.size(); }


  /* Modifiers */

  /* Note does not free the associated memory. */
  void clear()
  { resize( 0 ); }

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

    for ( size_type i = m_length; i > pos; --i )
    {
      const size_type cur_pos = i - 1;
      m_array[ cur_pos + n ] = std::move( m_array[ cur_pos ] );
    }

    for ( size_type i = 0; i < n; ++i )
    {
      const size_type cur_pos = pos + i;
      new ( &m_array[ cur_pos ] ) T();
    }

    m_length = new_length;
  }

  iterator insert( const_iterator pos, const T& value )
  {
    const size_type index = pos - begin();
    emplace( 1, index );
    m_array[ index ] = value();
    return begin() + index;
  }

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

  void push_back( const_reference value )
  {
    m_length++;
    if ( m_length > capacity() )
    {
      dynamicRealloc( m_length );
    }

    new ( &m_array[ m_length - 1] ) T(value);
  }

  void push_back( rvalue_reference value )
  {
    m_length++;
    if ( m_length > capacity() )
    {
      dynamicRealloc( m_length );
    }

    new ( &m_array[ m_length - 1] ) T( std::move( value ) );
  }

  void pop_back()
  { erase( end() ); }

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


  ChaiVector<T> deep_copy() const
  { 
    return ChaiVector( chai::deepCopy( m_array ), m_length );
  }

private:

  ChaiVector( chai::ManagedArray<T>&& source, size_type length ) :
    m_array( std::move( source ) ),
    m_length( length ),
    m_copied( false )
  {}

  void realloc( size_type new_capacity )
  {
    const size_type initial_capacity = capacity();
    if ( capacity() == 0 )
    {
      m_array.allocate( new_capacity );
    }
    else
    {
      m_array.reallocate( new_capacity );
    }
  }

  void dynamicRealloc( size_type new_length )
  { reserve( 2 * new_length ); }

  chai::ManagedArray<T> m_array;
  size_type m_length;
  bool m_copied;
};
