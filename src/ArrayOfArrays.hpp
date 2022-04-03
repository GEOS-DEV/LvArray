/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file ArrayOfArrays.hpp
 * @brief Contains the implementation of LvArray::ArrayOfArrays.
 */

#pragma once

#include "ArrayOfArraysView.hpp"

namespace LvArray
{

// Forward declaration of the ArrayOfSets class so that we can define the assimilate method.
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
class ArrayOfSets;

/**
 * @class ArrayOfArrays
 * @brief This class implements an array of arrays like object with contiguous storage.
 * @tparam T the type stored in the arrays.
 * @tparam INDEX_TYPE the integer to use for indexing.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayOfArrays : protected ArrayOfArraysView< T, INDEX_TYPE, false, BUFFER_TYPE >
{
  /// An alias for the parent class.
  using ParentClass = ArrayOfArraysView< T, INDEX_TYPE, false, BUFFER_TYPE >;

public:
  using typename ParentClass::ValueType;
  using typename ParentClass::IndexType;
  using typename ParentClass::value_type;
  using typename ParentClass::size_type;

  /**
   * @name Constructors and destructor.
   */
  ///@{

  /**
   * @brief Constructor.
   * @param numArrays the number of arrays.
   * @param defaultArrayCapacity the initial capacity of each array.
   */
  inline
  ArrayOfArrays( INDEX_TYPE const numArrays=0, INDEX_TYPE const defaultArrayCapacity=0 ):
    ParentClass( true )
  {
    resize( numArrays, defaultArrayCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param src the ArrayOfArrays to copy.
   */
  inline
  ArrayOfArrays( ArrayOfArrays const & src ):
    ParentClass( true )
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   */
  inline
  ArrayOfArrays( ArrayOfArrays && ) = default;

  /**
   * @brief Destructor, frees the values, sizes and offsets buffers.
   */
  ~ArrayOfArrays()
  { ParentClass::free(); }

  ///@}

  /**
   * @name ArrayOfArraysView creation methods
   */
  ///@{

  /**
   * @copydoc ParentClass::toView
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, false, BUFFER_TYPE >
  toView() const &
  { return ParentClass::toView(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArrayOfArraysView.
   * @note This cannot be called on a rvalue since the @c ArrayOfArraysView would
   *   contain the buffer of the current @c ArrayOfArrays that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, false, BUFFER_TYPE >
  toView() const && = delete;

  /**
   * @copydoc ParentClass::toViewConstSizes
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConstSizes() const &
  { return ParentClass::toViewConstSizes(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArrayOfArraysView.
   * @note This cannot be called on a rvalue since the @c ArrayOfArraysView would
   *   contain the buffer of the current @c ArrayOfArrays that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConstSizes() const && = delete;

  /**
   * @copydoc ParentClass::toViewConst
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConst() const &
  { return ParentClass::toViewConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArrayOfArraysView.
   * @note This cannot be called on a rvalue since the @c ArrayOfArraysView would
   *   contain the buffer of the current @c ArrayOfArrays that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConst() const && = delete;

  ///@}

  /**
   * @name Methods to construct the array from scratch.
   */
  ///@{

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the ArrayOfArrays to copy.
   * @return *this.
   */
  inline
  ArrayOfArrays & operator=( ArrayOfArrays const & src )
  {
    ParentClass::setEqualTo( src.m_numArrays,
                             src.m_offsets[ src.m_numArrays ],
                             src.m_offsets,
                             src.m_sizes,
                             src.m_values );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param src The ArrayOfArrays to be moved from.
   * @return *this
   */
  inline
  ArrayOfArrays & operator=( ArrayOfArrays && src )
  {
    ParentClass::free();
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  /**
   * @brief Steal the resources from an ArrayOfSets and convert it to an ArrayOfArrays.
   * @param src the ArrayOfSets to convert.
   */
  inline
  void assimilate( ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > && src )
  {
    ParentClass::free();
    ParentClass::assimilate( reinterpret_cast< ParentClass && >( src ) );
  }

  using ParentClass::resizeFromCapacities;
  using ParentClass::resizeFromOffsets;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @copydoc ParentClass::size
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  inline
  INDEX_TYPE size() const
  { return ParentClass::size(); }

  using ParentClass::sizeOfArray;
  using ParentClass::capacity;
  using ParentClass::capacityOfArray;
  using ParentClass::valueCapacity;

  ///@}

  /**
   * @name Methods that provide access to the data
   */
  ///@{

  using ParentClass::operator[];
  using ParentClass::operator();

  ///@}

  /**
   * @name Methods to modify the outer array.
   */
  ///@{

  using ParentClass::reserve;
  using ParentClass::reserveValues;

  /**
   * @copydoc ArrayOfArraysView::resize
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void resize( INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity=0 )
  { return ParentClass::resize( newSize, defaultArrayCapacity ); }

  using ParentClass::compress;

  ///@}

  /**
   * @name Methods to create or remove an inner array.
   */
  ///@{

  /**
   * @brief Append an array.
   * @param n the size of the array.
   */
  void appendArray( INDEX_TYPE const n )
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( n ) );

    INDEX_TYPE const maxOffset = this->m_offsets[ this->m_numArrays ];
    bufferManipulation::emplaceBack( this->m_offsets, this->m_numArrays + 1, maxOffset );
    bufferManipulation::emplaceBack( this->m_sizes, this->m_numArrays, 0 );
    ++this->m_numArrays;

    resizeArray( this->m_numArrays - 1, n );
  }

  /**
   * @brief Append an array.
   * @tparam ITER An iterator, the type of @p first and @p last.
   * @param first An iterator to the first value of the array to append.
   * @param last An iterator to the end of the array to append.
   */
  template< typename ITER >
  void appendArray( ITER const first, ITER const last )
  {
    INDEX_TYPE const maxOffset = this->m_offsets[ this->m_numArrays ];
    bufferManipulation::emplaceBack( this->m_offsets, this->m_numArrays + 1, maxOffset );
    bufferManipulation::emplaceBack( this->m_sizes, this->m_numArrays, 0 );
    ++this->m_numArrays;

    appendToArray( this->m_numArrays - 1, first, last );
  }

  /**
   * @brief Insert an array.
   * @tparam ITER An iterator, the type of @p first and @p last.
   * @param i the position to insert the array.
   * @param first An iterator to the first value of the array to insert.
   * @param last An iterator to the end of the array to insert.
   */
  template< typename ITER >
  void insertArray( INDEX_TYPE const i, ITER const first, ITER const last )
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i );

    // Insert an array of capacity zero at the given location
    INDEX_TYPE const offset = this->m_offsets[ i ];
    bufferManipulation::emplace( this->m_offsets, this->m_numArrays + 1, i + 1, offset );
    bufferManipulation::emplace( this->m_sizes, this->m_numArrays, i, 0 );
    ++this->m_numArrays;

    // Append to the new array
    appendToArray( i, first, last );
  }

  /**
   * @brief Erase an array.
   * @param i the position of the array to erase.
   */
  void eraseArray( INDEX_TYPE const i )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    setCapacityOfArray( i, 0 );
    bufferManipulation::erase( this->m_offsets, this->m_numArrays + 1, i + 1 );
    bufferManipulation::erase( this->m_sizes, this->m_numArrays, i );
    --this->m_numArrays;
  }

  ///@}

  /**
   * @name Methods to modify an inner array.
   */
  ///@{

  /**
   * @brief Append a value to an array constructing it in place with the given arguments.
   * @tparam ARGS Variadic pack of types used to construct T.
   * @param i the array to append to.
   * @param args A variadic pack of arguments that are forwarded to construct the new value.
   */
  template< typename ... ARGS >
  void emplaceBack( INDEX_TYPE const i, ARGS && ... args )
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::emplaceBack( i, std::forward< ARGS >( args )... );
  }

  /**
   * @brief Append values to an array.
   * @tparam ITER An iterator, the type of @p first and @p last.
   * @param i the array to append to.
   * @param first An iterator to the first value to append.
   * @param last An iterator to the end of the values to append.
   */
  template< typename ITER >
  void appendToArray( INDEX_TYPE const i, ITER const first, ITER const last )
  {
    INDEX_TYPE const n = arrayManipulation::iterDistance( first, last );
    dynamicallyGrowArray( i, n );
    ParentClass::appendToArray( i, first, last );
  }

  using ParentClass::emplaceBackAtomic;

  /**
   * @brief Insert a value into an array constructing it in place.
   * @tparam ARGS Variadic pack of types used to construct T.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param args A variadic pack of arguments that are forwarded to construct the new value.
   */
  template< typename ... ARGS >
  void emplace( INDEX_TYPE const i, INDEX_TYPE const j, ARGS && ... args )
  {
    dynamicallyGrowArray( i, 1 );
    ParentClass::emplace( i, j, std::forward< ARGS >( args )... );
  }

  /**
   * @brief Insert values into an array.
   * @tparam ITER An iterator, the type of @p first and @p last.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the end of the values to insert.
   */
  template< typename ITER >
  void insertIntoArray( INDEX_TYPE const i, INDEX_TYPE const j, ITER const first, ITER const last )
  {
    INDEX_TYPE const n = arrayManipulation::iterDistance( first, last );
    dynamicallyGrowArray( i, n );
    ParentClass::insertIntoArray( i, j, first, last );
  }

  using ParentClass::eraseFromArray;

  /**
   * @brief Set the number of values in an array.
   * @tparam ARGS variadic template parameter of the types used to initialize any new values with.
   * @param i the array to resize.
   * @param newSize the value to set the size of the array to.
   * @param args variadic parameter pack of the arguments used to initialize any new values with.
   */
  template< class ... ARGS >
  void resizeArray( INDEX_TYPE const i, INDEX_TYPE const newSize, ARGS && ... args )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( newSize ) );

    if( newSize > capacityOfArray( i ) )
    {
      setCapacityOfArray( i, newSize );
    }

    INDEX_TYPE const prevSize = sizeOfArray( i );
    T * const values = (*this)[i];
    arrayManipulation::resize( values, prevSize, newSize, std::forward< ARGS >( args )... );
    this->m_sizes[ i ] = newSize;
  }

  /**
   * @brief Clear the given array.
   * @param i The index of the array to clear.
   */
  void clearArray( INDEX_TYPE const i )
  { resizeArray( i, 0 ); }

  /**
   * @brief Set the capacity of an array.
   * @param i the array to set the capacity of.
   * @param newCapacity the value to set the capacity of the array to.
   */
  void setCapacityOfArray( INDEX_TYPE const i, INDEX_TYPE const newCapacity )
  { ParentClass::setCapacityOfArray( i, newCapacity ); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc ArrayOfArraysView::move
   * @note This is just a wrapper around the ArrayOfArraysView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool touch=true ) const
  { ParentClass::move( space, touch ); }

  ///@}

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name The name to display.
   */
  void setName( std::string const & name )
  { ParentClass::template setName< decltype( *this ) >( name ); }

private:

  /**
   * @brief Dynamically grow the capacity of an array.
   * @param i the array to grow.
   * @param increase the increase in the size of the array.
   */
  void dynamicallyGrowArray( INDEX_TYPE const i, INDEX_TYPE const increase )
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( increase ) );

    INDEX_TYPE const newArraySize = sizeOfArray( i ) + increase;
    if( newArraySize > capacityOfArray( i ))
    {
      setCapacityOfArray( i, 2 * newArraySize );
    }
  }
};

} /* namespace LvArray */
