/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file ArrayOfArraysView.hpp
 * @brief Contains the implementation of LvArray::ArrayOfArraysView.
 */

#pragma once

// Source includes
#include "bufferManipulation.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "typeManipulation.hpp"
#include "math.hpp"
#include "umpireInterface.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <cstring>

#ifdef LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p i is a valid array index.
 * @param i The array index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_BOUNDS( i ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size(), \
                    "Bounds Check Failed: i=" << i << " size()=" << this->size() )

/**
 * @brief Check that @p i is a valid array index and that @p j is a valid index into that array.
 * @param i The array index to check.
 * @param j The index into the array to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_BOUNDS2( i, j ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size() || \
                    !arrayManipulation::isPositive( j ) || j >= this->m_sizes[ i ], \
                    "Bounds Check Failed: i=" << i << " size()=" << this->size() << \
                    " j=" << j << " m_sizes[ i ]=" << this->m_sizes[ i ] )

/**
 * @brief Check that @p i is a valid index to insert an array at.
 * @param i The array index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( i ) || i > this->size(), \
                    "Insert Bounds Check Failed: i=" << i << " size()=" << this->size() )

/**
 * @brief Check that @p i is a valid array index and that @p j is a valid insertion index into that array.
 * @param i The array index to check.
 * @param j The index into the array to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( i ) || i >= this->size() || \
                    !arrayManipulation::isPositive( j ) || j > this->sizeOfArray( i ), \
                    "Insert Bounds Check Failed: i=" << i << " size()=" << this->size() << \
                    " j=" << j << " sizeOfArray( i )=" << this->sizeOfArray( i ) )

/**
 * @brief Check that the capacity of array @p i isn't exceeded when the size is increased by @p increase.
 * @param i The array index to check.
 * @param increase The increases in the capacity.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CAPACITY_CHECK( i, increase ) \
  LVARRAY_ERROR_IF( this->sizeOfArray( i ) + increase > this->capacityOfArray( i ), \
                    "Capacity Check Failed: i=" << i << " increase=" << increase << \
                    " sizeOfArray( i )=" << this->sizeOfArray( i ) << " capacityOfArray( i )=" << \
                    this->capacityOfArray( i ) )

/**
 * @brief Check that the capacity of array @p i isn't exceeded when the size is increased by @p increase.
 * @param i The array index to check.
 * @param previousSize The previous size of the array.
 * @param increase The increases in the capacity.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, increase ) \
  LVARRAY_ERROR_IF( previousSize + increase > this->capacityOfArray( i ), \
                    "Capacity Check Failed: i=" << i << " increase=" << increase << \
                    " sizeOfArray( i )=" << previousSize << " capacityOfArray( i )=" << \
                    this->capacityOfArray( i ) )

#else // LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p i is a valid array index.
 * @param i The array index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_BOUNDS( i )

/**
 * @brief Check that @p i is a valid array index and that @p j is a valid index into that array.
 * @param i The array index to check.
 * @param j The index into the array to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_BOUNDS2( i, j )

/**
 * @brief Check that @p i is a valid index to insert an array at.
 * @param i The array index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i )

/**
 * @brief Check that @p i is a valid array index and that @p j is a valid insertion index into that array.
 * @param i The array index to check.
 * @param j The index into the array to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j )

/**
 * @brief Check that the capacity of array @p i isn't exceeded when the size is increased by @p increase.
 * @param i The array index to check.
 * @param increase The increases in the capacity.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_CAPACITY_CHECK( i, increase )

/**
 * @brief Check that the capacity of array @p i isn't exceeded when the size is increased by @p increase.
 * @param i The array index to check.
 * @param previousSize The previous size of the array.
 * @param increase The increases in the capacity.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, increase )

#endif // LVARRAY_BOUNDS_CHECK

namespace LvArray
{

/**
 * @class ArrayOfArraysView
 * @brief This class provides a view into an array of arrays like object.
 * @tparam T the type stored in the arrays.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam CONST_SIZES true iff the size of each array is constant.
 *
 * When INDEX_TYPE is const m_offsets is not copied between memory spaces.
 * When accessing this class directly (not through an ArrayOfArrays object)
 * INDEX_TYPE should always be const since ArrayOfArraysView is not allowed to modify the offsets.
 *
 * When CONST_SIZES is true m_sizes is not copied between memory spaces.
 */
template< typename T,
          typename INDEX_TYPE,
          bool CONST_SIZES,
          template< typename > class BUFFER_TYPE >
class ArrayOfArraysView
{
protected:
  /// Since INDEX_TYPE should always be const we need an alias for the non const version.
  using INDEX_TYPE_NC = std::remove_const_t< INDEX_TYPE >;

  /// The type contained by the m_sizes buffer.
  using SIZE_TYPE = std::conditional_t< CONST_SIZES, INDEX_TYPE const, INDEX_TYPE_NC >;

public:
  static_assert( !std::is_const< T >::value || (std::is_const< INDEX_TYPE >::value && CONST_SIZES),
                 "When T is const INDEX_TYPE must also be const and CONST_SIZES must be true" );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );

  /// An alias for the type contained in the inner arrays.
  using ValueType = T;

  /// The integer type used for indexing.
  using IndexType = INDEX_TYPE;

  /// An alias for the type contained in the inner arrays, here for stl compatability.
  using value_type = T;

  /// The integer type used for indexing, here for stl compatability.
  using size_type = INDEX_TYPE;

  /**
   * @name Constructors, destructor and assignment operators
   */
  ///@{

  /**
   * @brief A constructor to create an uninitialized ArrayOfArraysView.
   * @note An uninitialized ArrayOfArraysView should not be used until it is assigned to.
   */
  ArrayOfArraysView() = default;

  /**
   * @brief Default copy constructor.
   * @note The copy constructor will trigger the copy constructor for @tparam BUFFER_TYPE
   */
  ArrayOfArraysView( ArrayOfArraysView const & ) = default;

  /**
   * @brief Default move constructor.
   * @param src the ArrayOfArraysView to be moved from.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView( ArrayOfArraysView && src ):
    m_numArrays( src.m_numArrays ),
    m_offsets( std::move( src.m_offsets ) ),
    m_sizes( std::move( src.m_sizes ) ),
    m_values( std::move( src.m_values ) )
  { src.m_numArrays = 0; }

  /**
   * @brief Construct a new ArrayOfArraysView from the given buffers.
   * @param numArrays The number of arrays.
   * @param offsets The offsets buffer, of size @p numArrays + 1.
   * @param sizes The sizes buffer, of size @p numArrays.
   * @param values The values buffer, of size @p offsets[ numArrays ].
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView( INDEX_TYPE const numArrays,
                     BUFFER_TYPE< INDEX_TYPE > const & offsets,
                     BUFFER_TYPE< SIZE_TYPE > const & sizes,
                     BUFFER_TYPE< T > const & values ):
    m_numArrays( numArrays ),
    m_offsets( offsets ),
    m_sizes( sizes ),
    m_values( values )
  {}

  /**
   * @brief Default copy assignment operator.
   * @return *this.
   */
  inline
  ArrayOfArraysView & operator=( ArrayOfArraysView const & ) = default;

  /**
   * @brief Move assignment operator..
   * @param src the SparsityPatternView to be moved from.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  inline
  ArrayOfArraysView & operator=( ArrayOfArraysView && src )
  {
    m_numArrays = src.m_numArrays;
    src.m_numArrays = 0;
    m_offsets = std::move( src.m_offsets );
    m_sizes = std::move( src.m_sizes );
    m_values = std::move( src.m_values );
    return *this;
  }

  ///@}

  /**
   * @name ArrayOfArraysView creation methods
   */
  ///@{

  /**
   * @return Return a new ArrayOfArraysView<T, INDEX_TYPE const, CONST_SIZES>.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, CONST_SIZES, BUFFER_TYPE >
  toView() const
  {
    return ArrayOfArraysView< T, INDEX_TYPE const, CONST_SIZES, BUFFER_TYPE >( size(),
                                                                               this->m_offsets,
                                                                               this->m_sizes,
                                                                               this->m_values );
  }

  /**
   * @return Return a new ArrayOfArraysView<T, INDEX_TYPE const, true>.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConstSizes() const
  {
    return ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >( size(),
                                                                        this->m_offsets,
                                                                        this->m_sizes,
                                                                        this->m_values );
  }

  /**
   * @return Return a new ArrayOfArraysView<T const, INDEX_TYPE const, true>.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >
  toViewConst() const
  {
    return ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >( size(),
                                                                              this->m_offsets,
                                                                              this->m_sizes,
                                                                              this->m_values );
  }

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @return Return the number of arrays.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC size() const
  { return m_numArrays; }

  /**
   * @return Return the size of the given array.
   * @param i the array to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  INDEX_TYPE_NC sizeOfArray( INDEX_TYPE const i ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return m_sizes[ i ];
  }

  /**
   * @return Return a pointer to the sizes of each outer array.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SIZE_TYPE const * getSizes() const
  {
    return m_sizes.data();
  }


  /**
   * @return Return a pointer to the value array offsets for each outer array
   *   with an entry for the end of the values.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE const * getOffsets() const
  {
    return m_offsets.data();
  }

  /**
   * @return Return a pointer to the values data.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  T const * getValues() const
  {
    return m_values.data();
  }



  /**
   * @return Return the number of (zero length) arrays that can be stored before reallocation.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITH_NDEBUG inline
  INDEX_TYPE_NC capacity() const
  {
    LVARRAY_ASSERT( m_sizes.capacity() < m_offsets.capacity());
    return m_sizes.capacity();
  }

  /**
   * @return Return the capacity of the given array.
   * @param i the array to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  INDEX_TYPE_NC capacityOfArray( INDEX_TYPE const i ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return m_offsets[ i + 1 ] - m_offsets[ i ];
  }

  /**
   * @return Return the total number values that can be stored before reallocation.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC valueCapacity() const
  { return m_values.capacity(); }

  ///@}

  /**
   * @name Methods that provide access to the data
   */
  ///@{

  /**
   * @return Return an ArraySlice1d to the values of the given array.
   * @param i the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  ArraySlice< T, 1, 0, INDEX_TYPE_NC > operator[]( INDEX_TYPE const i ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    return ArraySlice< T, 1, 0, INDEX_TYPE_NC >( m_values.data() + m_offsets[ i ], &m_sizes[ i ], nullptr );
  }

  /**
   * @return Return a reference to the value at the given position in the given array.
   * @param i the array to access.
   * @param j the index within the array to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  T & operator()( INDEX_TYPE const i, INDEX_TYPE const j ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS2( i, j );
    return m_values[m_offsets[ i ] + j];
  }

  ///@}

  /**
   * @name Methods that modify the size of an inner array.
   */
  ///@{

  /**
   * @brief Append a value to an array.
   * @tparam ARGS A variadic pack of types used to construct the new T, the types of @p args.
   * @param i the array to append to.
   * @param args The variadic pack of arguments forwared to construct the new value.
   * @pre Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given array has enough space for the new values.
   */
  template< typename ... ARGS >
  LVARRAY_HOST_DEVICE inline
  void emplaceBack( INDEX_TYPE const i, ARGS && ... args ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::emplaceBack( ptr, sizeOfArray( i ), std::forward< ARGS >( args ) ... );
    m_sizes[ i ] += 1;
  }

  /**
   * @brief Append a value to an array in a thread safe manner.
   * @tparam POLICY The RAJA atomic policy to use to increment the size of the array.
   * @tparam ARGS A variadic pack of types used to construct the new T, the types of @p args.
   * @param i the array to append to.
   * @param args The variadic pack of arguments forwared to construct the new value.
   * @pre Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given array has enough space for the new values.
   */
  template< typename POLICY, typename ... ARGS >
  LVARRAY_HOST_DEVICE inline
  void emplaceBackAtomic( INDEX_TYPE const i, ARGS && ... args ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    T * const ptr = m_values.data() + m_offsets[ i ];
    INDEX_TYPE const previousSize = RAJA::atomicInc< POLICY >( &m_sizes[ i ] );
    ARRAYOFARRAYS_ATOMIC_CAPACITY_CHECK( i, previousSize, 1 );

    arrayManipulation::emplaceBack( ptr, previousSize, std::forward< ARGS >( args ) ... );
  }

  /**
   * @brief Append values to an array.
   * @tparam ITER An iterator, they type of @p first and @p last.
   * @param i the array to append to.
   * @param first An iterator to the first value to append.
   * @param last An iterator to the end of the values to append.
   * @pre Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given array has enough space for the new values.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  void appendToArray( INDEX_TYPE const i, ITER const first, ITER const last ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    T * const ptr = m_values.data() + m_offsets[ i ];
    INDEX_TYPE const n = arrayManipulation::append( ptr, sizeOfArray( i ), first, last );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, n );
    m_sizes[ i ] += n;
  }

  /**
   * @brief Insert a value into an array.
   * @tparam ARGS A variadic pack of types used to construct the new T, the types of @p args.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param args The variadic pack of arguments forwared to construct the new value.
   * @pre Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given array has enough space for the new values.
   */
  template< typename ... ARGS >
  LVARRAY_HOST_DEVICE inline
  void emplace( INDEX_TYPE const i, INDEX_TYPE const j, ARGS && ... args ) const
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, 1 );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::emplace( ptr, sizeOfArray( i ), j, std::forward< ARGS >( args )... );
    m_sizes[ i ]++;
  }

  /**
   * @brief Insert values into an array.
   * @tparam ITER An iterator, they type of @p first and @p last.
   * @param i the array to insert into.
   * @param j the position at which to insert.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the end of the values to insert.
   * @pre Since the ArrayOfArraysView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given array has enough space for the new values.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  void insertIntoArray( INDEX_TYPE const i,
                        INDEX_TYPE const j,
                        ITER const first,
                        ITER const last ) const
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS2( i, j );
    INDEX_TYPE const n = arrayManipulation::iterDistance( first, last );
    ARRAYOFARRAYS_CAPACITY_CHECK( i, n );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::insert( ptr, sizeOfArray( i ), j, first, n );
    m_sizes[ i ] += n;
  }

  /**
   * @brief Erase values from an array.
   * @param i the array to erase values from.
   * @param j the position at which to begin erasing.
   * @param n the number of values to erase.
   */
  LVARRAY_HOST_DEVICE inline
  void eraseFromArray( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const n=1 ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS2( i, j );

    T * const ptr = m_values.data() + m_offsets[ i ];
    arrayManipulation::erase( ptr, sizeOfArray( i ), j, n );
    m_sizes[ i ] -= n;
  }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @brief Touch the memory in @p space.
   * @param space The memory space in which a touch will be recorded.
   */
  void registerTouch( MemorySpace const space ) const
  {
    m_values.registerTouch( space );
    m_sizes.registerTouch( space );
    m_offsets.registerTouch( space );
  }

  /**
   * @brief Move this ArrayOfArrays to the given memory space.
   * @param space The memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   */
  void move( MemorySpace const space, bool touch=true ) const
  {
    m_values.move( space, touch );
    m_sizes.move( space, touch );

  #if defined(LVARRAY_USE_CUDA)
    if( space == MemorySpace::cuda ) touch = false;
  #endif
  #if defined(LVARRAY_USE_HIP)
    if( space == MemorySpace::hip ) touch = false;
  #endif
    m_offsets.move( space, touch );
  }


  ///@}

protected:

  /**
   * @brief Alias for a std::pair of buffers.
   * @tparam U The type contained in the buffers.
   */
  template< typename U >
  using PairOfBuffers = std::pair< BUFFER_TYPE< U > &, BUFFER_TYPE< U > const & >;

  /**
   * @brief Protected constructor to be used by parent classes.
   * @note The unused boolean parameter is to distinguish this from the default constructor.
   */
  ArrayOfArraysView( bool ):
    m_numArrays( 0 ),
    m_offsets( true ),
    m_sizes( true ),
    m_values( true )
  {}

  /**
   * @name Methods intended to be publically aliased directly in a derived array class.
   */
  ///@{

  /**
   * @brief Set the number of arrays.
   * @param newSize the new number of arrays.
   * @param defaultArrayCapacity the default capacity for each new array.
   */
  void resize( INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity=0 )
  { return resizeImpl( newSize, defaultArrayCapacity ); }

  /**
   * @brief Reserve space for the given number of arrays.
   * @param newCapacity the new minimum capacity for the number of arrays.
   */
  void reserve( INDEX_TYPE const newCapacity )
  {
    bufferManipulation::reserve( m_offsets, m_numArrays + 1, MemorySpace::host, newCapacity + 1 );
    bufferManipulation::reserve( m_sizes, m_numArrays, MemorySpace::host, newCapacity );
  }

  /**
   * @brief Reserve space for the given number of values.
   * @tparam BUFFERS variadic template where each type is a BUFFER_TYPE.
   * @param newValueCapacity the new minimum capacity for the number of values across all arrays.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *        similarly to m_values.
   */
  template< class ... BUFFERS >
  void reserveValues( INDEX_TYPE const newValueCapacity, BUFFERS & ... buffers )
  {
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    typeManipulation::forEachArg( [newValueCapacity, maxOffset] ( auto & buffer )
    {
      bufferManipulation::reserve( buffer, maxOffset, MemorySpace::host, newValueCapacity );
    }, m_values, buffers ... );
  }

  /**
   * @brief Compress the arrays so that the values of each array are contiguous with no extra capacity in between.
   * @tparam BUFFERS variadic template where each type is a BUFFER_TYPE.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *   similarly to m_values.
   * @note This method doesn't free any memory.
   */
  template< class ... BUFFERS >
  void compress( BUFFERS & ... buffers )
  {
    if( m_numArrays == 0 ) return;

    for( INDEX_TYPE i = 0; i < m_numArrays - 1; ++i )
    {
      INDEX_TYPE const nextOffset = m_offsets[ i + 1 ];
      INDEX_TYPE const shiftAmount = nextOffset - m_offsets[ i ] - sizeOfArray( i );
      INDEX_TYPE const sizeOfNextArray = sizeOfArray( i + 1 );

      // Shift the values in the next array down.
      typeManipulation::forEachArg(
        [sizeOfNextArray, nextOffset, shiftAmount] ( auto & buffer )
      {
        arrayManipulation::uninitializedShiftDown( &buffer[ nextOffset ], sizeOfNextArray, shiftAmount );
      }, m_values, buffers ... );

      // And update the offsets.
      m_offsets[ i + 1 ] -= shiftAmount;
    }

    // Update the last offset.
    m_offsets[ m_numArrays ] = m_offsets[ m_numArrays - 1 ] + sizeOfArray( m_numArrays - 1 );
  }

  /**
   * @brief Clears the array and creates a new array with the given number of sub-arrays.
   * @param numSubArrays The new number of arrays.
   * @param offsets A pointer to an array of length @p numSubArrays+1 containing the offset
   *   of each new sub array. Offsets are precomputed by the caller.
   * @param buffers A variadic pack of buffers to treat similarly to m_values.
   */
  template< typename ... BUFFERS >
  void resizeFromOffsets( INDEX_TYPE const numSubArrays,
                          INDEX_TYPE const * const offsets,
                          BUFFERS & ... buffers )
  {
    auto const fillOffsets = [&]()
    {
      arrayManipulation::uninitializedCopy( offsets,
                                            offsets + numSubArrays + 1,
                                            m_offsets.data() );
    };
    resizeFromOffsetsImpl( numSubArrays, fillOffsets, buffers ... );
  }

  /**
   * @tparam POLICY The RAJA policy used to convert @p capacities into the offsets array.
   *   Should NOT be a device policy.
   * @brief Clears the array and creates a new array with the given number of sub-arrays.
   * @param numSubArrays The new number of arrays.
   * @param capacities A pointer to an array of length @p numSubArrays containing the capacity
   *   of each new sub array.
   * @param buffers A variadic pack of buffers to treat similarly to m_values.
   */
  template< typename POLICY, typename ... BUFFERS >
  void resizeFromCapacities( INDEX_TYPE const numSubArrays,
                             INDEX_TYPE const * const capacities,
                             BUFFERS & ... buffers )
  {
    auto const fillOffsets = [&]()
    {
      m_offsets[ 0 ] = 0;
      RAJA::inclusive_scan< POLICY >( RAJA::make_span< INDEX_TYPE const * >( capacities, numSubArrays ),
                                      RAJA::make_span< INDEX_TYPE * >( m_offsets.data()+1, numSubArrays ) );
    };
    resizeFromOffsetsImpl( numSubArrays, fillOffsets, buffers ... );
  }

  ///@}

  /**
   * @name Methods to be used to help implement a derived classes
   */
  ///@{

  /**
   * @brief Steal the resources of @p src, clearing it in the process.
   * @param src The ArrayOfArraysView to steal from.
   */
  void assimilate( ArrayOfArraysView< T, INDEX_TYPE, CONST_SIZES, BUFFER_TYPE > && src )
  { *this = std::move( src ); }

  /**
   * @copydoc resize
   * @tparam BUFFERS variadic template where each type is BUFFER_TYPE.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *   similarly to m_values.
   */
  template< typename ... BUFFERS >
  void resizeImpl( INDEX_TYPE const newSize, INDEX_TYPE const defaultArrayCapacity, BUFFERS & ... buffers )
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( newSize ) );

    INDEX_TYPE const offsetsSize = ( m_numArrays == 0 ) ? 0 : m_numArrays + 1;

    if( newSize < m_numArrays )
    {
      destroyValues( newSize, m_numArrays, buffers ... );
      bufferManipulation::resize( m_offsets, offsetsSize, newSize + 1, 0 );
      bufferManipulation::resize( m_sizes, m_numArrays, newSize, 0 );
    }
    else
    {
      // The ternary here accounts for the case where m_offsets hasn't been allocated yet (when calling from a
      // constructor).
      INDEX_TYPE const originalOffset = (m_numArrays == 0) ? 0 : m_offsets[m_numArrays];
      bufferManipulation::resize( m_offsets, offsetsSize, newSize + 1, originalOffset );
      bufferManipulation::resize( m_sizes, m_numArrays, newSize, 0 );

      if( defaultArrayCapacity > 0 )
      {
        for( INDEX_TYPE i = 1; i < newSize + 1 - m_numArrays; ++i )
        {
          m_offsets[ m_numArrays + i ] = originalOffset + i * defaultArrayCapacity;
        }

        INDEX_TYPE const totalSize = m_offsets[ newSize ];

        INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
        typeManipulation::forEachArg( [totalSize, maxOffset]( auto & buffer )
        {
          bufferManipulation::reserve( buffer, maxOffset, MemorySpace::host, totalSize );
        }, m_values, buffers ... );
      }
    }

    m_numArrays = newSize;
  }

  /**
   * @brief Destroy all the objects held by this array and free all associated memory.
   * @tparam BUFFERS variadic template where each type is a BUFFER_TYPE.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *        similarly to m_values.
   */
  template< class ... BUFFERS >
  void free( BUFFERS & ... buffers )
  {
    destroyValues( 0, m_numArrays, buffers ... );

    typeManipulation::forEachArg( []( auto & buffer )
    {
      buffer.free();
    }, m_sizes, m_offsets, m_values, buffers ... );

    m_numArrays = 0;
  }

  /**
   * @tparam PAIRS_OF_BUFFERS variadic template where each type is an PairOfBuffers.
   * @brief Set this ArrayOfArraysView equal to the provided arrays.
   * @param srcNumArrays The number of arrays in source.
   * @param srcMaxOffset The maximum offset in the source.
   * @param srcOffsets the source offsets array.
   * @param srcSizes the source sizes array.
   * @param srcValues the source values array.
   * @param pairs variadic parameter pack where each argument is an PairOfBuffers where each pair
   *   should be treated similarly to {m_values, srcValues}.
   */
  template< class ... PAIRS_OF_BUFFERS >
  void setEqualTo( INDEX_TYPE const srcNumArrays,
                   INDEX_TYPE const srcMaxOffset,
                   BUFFER_TYPE< INDEX_TYPE > const & srcOffsets,
                   BUFFER_TYPE< INDEX_TYPE > const & srcSizes,
                   BUFFER_TYPE< T > const & srcValues,
                   PAIRS_OF_BUFFERS && ... pairs )
  {
    destroyValues( 0, m_numArrays, pairs.first ... );

    INDEX_TYPE const offsetsSize = ( m_numArrays == 0 ) ? 0 : m_numArrays + 1;

    bufferManipulation::copyInto( m_offsets, offsetsSize, srcOffsets, srcNumArrays + 1 );
    bufferManipulation::copyInto( m_sizes, m_numArrays, srcSizes, srcNumArrays );

    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    typeManipulation::forEachArg( [maxOffset, srcMaxOffset]( auto & dstBuffer )
    {
      bufferManipulation::reserve( dstBuffer, maxOffset, MemorySpace::host, srcMaxOffset );
    }, m_values, pairs.first ... );

    m_numArrays = srcNumArrays;

    typeManipulation::forEachArg( [this] ( auto & pair )
    {
      auto & dstBuffer = pair.first;
      auto const & srcBuffer = pair.second;

      for( INDEX_TYPE_NC i = 0; i < m_numArrays; ++i )
      {
        INDEX_TYPE const offset = m_offsets[ i ];
        INDEX_TYPE const arraySize = sizeOfArray( i );
        arrayManipulation::uninitializedCopy( &srcBuffer[ offset ],
                                              &srcBuffer[ offset ] + arraySize,
                                              &dstBuffer[ offset ] );
      }
    }, PairOfBuffers< T >( m_values, srcValues ), pairs ... );
  }

  /**
   * @brief Set the capacity of the given array.
   * @tparam BUFFERS variadic template where each type is a BUFFER_TYPE.
   * @param i the array to set the capacity of.
   * @param newCapacity the value to set the capacity of the given array to.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *   similarly to m_values.
   */
  template< class ... BUFFERS >
  void setCapacityOfArray( INDEX_TYPE const i, INDEX_TYPE const newCapacity, BUFFERS & ... buffers )
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    LVARRAY_ASSERT( arrayManipulation::isPositive( newCapacity ) );

    INDEX_TYPE const arrayCapacity = capacityOfArray( i );
    INDEX_TYPE const capacityIncrease = newCapacity - arrayCapacity;
    if( capacityIncrease == 0 ) return;

    if( capacityIncrease > 0 )
    {
      INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
      typeManipulation::forEachArg(
        [this, i, maxOffset, capacityIncrease]( auto & buffer )
      {
        INDEX_TYPE const totalCapacity = maxOffset + capacityIncrease;

        if( buffer.capacity() > totalCapacity )
        {
          // If the buffer has enough capacity then all we need to do is shift the sub-arrays around.
          for( INDEX_TYPE array = m_numArrays - 1; array > i; --array )
          {
            INDEX_TYPE const curArraySize = sizeOfArray( array );
            INDEX_TYPE const curArrayOffset = m_offsets[ array ];
            arrayManipulation::uninitializedShiftUp( &buffer[ curArrayOffset ], curArraySize, capacityIncrease );
          }
        }
        else
        {
          // Otherwise we create a new buffer with enough capacity and move the values over.
          // We create a new buffer to avoid moving from uninitialized values, this should also be faster.
          auto newBuffer = std::remove_reference_t< decltype( buffer ) >( true );
          bufferManipulation::dynamicReserve( newBuffer, 0, totalCapacity );

          for( INDEX_TYPE array = 0; array < m_numArrays; ++array )
          {
            INDEX_TYPE const curArraySize = sizeOfArray( array );
            INDEX_TYPE const curArrayOffset = m_offsets[ array ];
            INDEX_TYPE shift = array > i ? capacityIncrease : 0;
            arrayManipulation::uninitializedMove( &newBuffer[ curArrayOffset + shift ], curArraySize, &buffer[ curArrayOffset ] );
          }

          buffer = std::move( newBuffer );
        }
      },
        m_values, buffers ...
        );
    }
    else
    {
      INDEX_TYPE const arrayOffset = m_offsets[ i ];
      INDEX_TYPE const capacityDecrease = -capacityIncrease;

      INDEX_TYPE const prevArraySize = sizeOfArray( i );
      INDEX_TYPE const newArraySize = math::min( prevArraySize, newCapacity );

      m_sizes[ i ] = newArraySize;
      typeManipulation::forEachArg(
        [this, i, capacityDecrease, arrayOffset, newArraySize, prevArraySize] ( auto & buffer )
      {
        // Delete the values at the end of the array.
        arrayManipulation::destroy( &buffer[ arrayOffset + newArraySize ], prevArraySize - newArraySize );

        // Shift down the values of subsequent arrays.
        for( INDEX_TYPE array = i + 1; array < m_numArrays; ++array )
        {
          INDEX_TYPE const curArraySize = sizeOfArray( array );
          INDEX_TYPE const curArrayOffset = m_offsets[array];
          arrayManipulation::uninitializedShiftDown( &buffer[ curArrayOffset ], curArraySize, capacityDecrease );
        }
      },
        m_values, buffers ...
        );
    }

    // Update the offsets array
    for( INDEX_TYPE array = i + 1; array < m_numArrays + 1; ++array )
    {
      m_offsets[array] += capacityIncrease;
    }

    // We need to touch the offsets on the CPU because since it contains const data
    // when the ArrayOfArraysView gets copy constructed it doesn't get touched even though
    // it can then be modified via this method when called from a parent non-view class.
    m_offsets.registerTouch( MemorySpace::host );
  }

  /**
   * @tparam U They type of the owning object.
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name to display.
   */
  template< typename U >
  void setName( std::string const & name )
  {
    m_offsets.template setName< U >( name + "/m_offsets" );
    m_sizes.template setName< U >( name + "/m_sizes" );
    m_values.template setName< U >( name + "/m_values" );
  }

  ///@}

  /// The number of arrays contained.
  INDEX_TYPE_NC m_numArrays = 0;

  /// Holds the offset of each array, of length m_numArrays + 1. Array i begins at
  /// m_offsets[ i ] and has capacity m_offsets[i+1] - m_offsets[ i ].
  BUFFER_TYPE< INDEX_TYPE > m_offsets;

  /// Holds the size of each array.
  BUFFER_TYPE< SIZE_TYPE > m_sizes;

  /// Holds the values of each array. Values in the range [m_offsets[ i ], m_offsets[ i ] + m_sizes[ i ])
  /// are valid. All other entries contain uninitialized values.
  BUFFER_TYPE< T > m_values;

private:

  /**
   * @brief Destroy the values in arrays in the range [begin, end).
   * @tparam BUFFERS variadic template where each type is a BUFFER_TYPE.
   * @param begin the array to start with.
   * @param end where to stop destroying values.
   * @param buffers variadic parameter pack where each argument is a BUFFER_TYPE that should be treated
   *   similarly to m_values.
   * @note This is to be use by the non-view derived classes.
   * @note This doesn't free any memory.
   */
  template< class ... BUFFERS >
  void destroyValues( INDEX_TYPE const begin, INDEX_TYPE const end, BUFFERS & ... buffers )
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( begin );
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( end );

    // If the values aren't trivially destructable then the data needs to be moved back to the host.
    // This moves m_sizes and m_offsets, m_values and buffers are moved inside the loop.
    if( !typeManipulation::all_of_t< std::is_trivially_destructible< T >,
                                     std::is_trivially_destructible< typename BUFFERS::value_type > ... >::value )
    { move( MemorySpace::host, true ); }

    typeManipulation::forEachArg( [this, begin, end] ( auto & buffer )
    {
      if( !std::is_trivially_destructible< std::remove_reference_t< decltype( buffer[ 0 ] ) > >::value )
      {
        buffer.move( MemorySpace::host, true );
        for( INDEX_TYPE i = begin; i < end; ++i )
        {
          INDEX_TYPE const offset = m_offsets[ i ];
          INDEX_TYPE const arraySize = sizeOfArray( i );
          arrayManipulation::destroy( &buffer[ offset ], arraySize );
        }
      }
    }, m_values, buffers ... );
  }

  /**
   * @brief Clears the array and creates a new array with the given number of sub-arrays.
   * @param numSubArrays The new number of arrays.
   * @param fillOffsets A function that will be called to populate sub-array offsets.
   * @param buffers A variadic pack of buffers to treat similarly to m_values.
   * @note This is to be called by other resizeFrom functions.
   */
  template< typename FUNC, typename ... BUFFERS >
  void resizeFromOffsetsImpl( INDEX_TYPE const numSubArrays,
                              FUNC && fillOffsets,
                              BUFFERS & ... buffers )
  {
    LVARRAY_ASSERT( arrayManipulation::isPositive( numSubArrays ) );

    destroyValues( 0, m_numArrays, buffers ... );

    bufferManipulation::reserve( m_sizes, m_numArrays, MemorySpace::host, numSubArrays );
    umpireInterface::memset( m_sizes.data(), 0, m_sizes.capacity() * sizeof( INDEX_TYPE ) );

    INDEX_TYPE const offsetsSize = ( m_numArrays == 0 ) ? 0 : m_numArrays + 1;
    bufferManipulation::reserve( m_offsets, offsetsSize, MemorySpace::host, numSubArrays + 1 );

    fillOffsets();
    LVARRAY_ASSERT_EQ( m_offsets[0], 0 );
    LVARRAY_ASSERT( sortedArrayManipulation::isSorted( m_offsets.data(), m_offsets.data() + numSubArrays + 1 ) );

    m_numArrays = numSubArrays;
    INDEX_TYPE const maxOffset = m_offsets[ m_numArrays ];
    typeManipulation::forEachArg( [ maxOffset ] ( auto & buffer )
    {
      bufferManipulation::reserve( buffer, 0, MemorySpace::host, maxOffset );
    }, m_values, buffers ... );
  }
};

} /* namespace LvArray */
