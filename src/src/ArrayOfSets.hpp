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

/**
 * @file SparsityPattern.hpp
 */

#pragma once

#include "ArrayOfSetsView.hpp"

namespace LvArray
{

// Forward declaration of the ArrayOfArrays class so that we can define the stealFrom method.
template <class T, typename INDEX_TYPE>
class ArrayOfArrays;


template <class T, class INDEX_TYPE=std::ptrdiff_t>
class ArrayOfSets : protected ArrayOfSetsView<T, INDEX_TYPE>
{
public:

  // Aliasing public methods of ArrayOfSetsView.
  using ArrayOfSetsView<T, INDEX_TYPE>::toViewC;
  using ArrayOfSetsView<T, INDEX_TYPE>::toArrayOfArraysView;
  using ArrayOfSetsView<T, INDEX_TYPE>::capacity;
  using ArrayOfSetsView<T, INDEX_TYPE>::sizeOfSet;
  using ArrayOfSetsView<T, INDEX_TYPE>::capacityOfSet;
  using ArrayOfSetsView<T, INDEX_TYPE>::operator();
  using ArrayOfSetsView<T, INDEX_TYPE>::operator[];
  using ArrayOfSetsView<T, INDEX_TYPE>::getSetValues;

  using ArrayOfSetsView<T, INDEX_TYPE>::getIterableSet;
  using ArrayOfSetsView<T, INDEX_TYPE>::removeFromSet;
  using ArrayOfSetsView<T, INDEX_TYPE>::removeSortedFromSet;
  using ArrayOfSetsView<T, INDEX_TYPE>::contains;
  using ArrayOfSetsView<T, INDEX_TYPE>::setUserCallBack;
  using ArrayOfSetsView<T, INDEX_TYPE>::consistencyCheck;

  /**
   * @brief Return the number sets.
   * @note This needs is duplicated here for the intel compiler on cori. 
   */
  inline
  INDEX_TYPE size() const restrict_this
  { return m_sizes.size(); }

  inline
  ArrayOfSets(INDEX_TYPE const nsets=0, INDEX_TYPE defaultSetCapacity=0) restrict_this:
    ArrayOfSetsView<T, INDEX_TYPE>()
  { ArrayOfSetsView<T, INDEX_TYPE>::resize(nsets, defaultSetCapacity); }
  
  inline
  ArrayOfSets( ArrayOfSets const & src ) restrict_this:
    ArrayOfSetsView<T, INDEX_TYPE>()
  { *this = src; }

  inline
  ArrayOfSets( ArrayOfSets && src ) = default;

  inline
  ~ArrayOfSets() restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::free(); }

  CONSTEXPRFUNC inline
  operator ArrayOfSetsView<T, INDEX_TYPE const> const &
  () const restrict_this
  { return reinterpret_cast<ArrayOfSetsView<T, INDEX_TYPE const> const &>(*this); }

  inline
  ArrayOfSetsView<T, INDEX_TYPE const> const & toView() const restrict_this
  { return *this; }

  CONSTEXPRFUNC inline
  operator ArrayOfSetsView<T const, INDEX_TYPE const> const &
  () const restrict_this
  { return toViewC(); }

  template<class U=T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator ArrayOfArraysView<T const, INDEX_TYPE const, true>
  () const restrict_this
  { return toArrayOfArraysView(); }

  inline
  ArrayOfSets & operator=( ArrayOfSets const & src ) restrict_this
  {
    ArrayOfSetsView<T, INDEX_TYPE>::setEqualTo(src.m_offsets.toConst(), src.m_sizes.toConst(), src.m_values.toConst());
    return *this;
  }

  inline
  ArrayOfSets & operator=( ArrayOfSets && src ) = default;

  // This would be prime for omp parallelism.
  inline
  void stealFrom( ArrayOfArrays< T, INDEX_TYPE > && src, sortedArrayManipulation::Description const desc ) restrict_this
  {
    // Reinterpret cast to ArrayOfArraysView so that we don't have to include ArrayOfArrays.hpp.
    ArrayOfArraysView< T, INDEX_TYPE > && srcView = reinterpret_cast< ArrayOfArraysView< T, INDEX_TYPE > && >( src );
    m_offsets = std::move( srcView.m_offsets );
    m_sizes = std::move( srcView.m_sizes );
    m_values = std::move( srcView.m_values );

    INDEX_TYPE const numSets = size();
    if (desc == sortedArrayManipulation::UNSORTED_NO_DUPLICATES)
    {
      for ( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );
        std::sort( setValues, setValues + numValues );
      }
    }
    if (desc == sortedArrayManipulation::SORTED_WITH_DUPLICATES)
    {
      for ( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );

        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( setValues, numValues );
        arrayManipulation::resize( setValues, numValues, numUniqueValues );
        m_sizes[ i ] = numUniqueValues;
      }
    }
    if (desc == sortedArrayManipulation::UNSORTED_WITH_DUPLICATES)
    {
      for ( INDEX_TYPE i = 0; i < numSets; ++i )
      {
        T * const setValues = getSetValues( i );
        INDEX_TYPE const numValues = sizeOfSet( i );
        std::sort( setValues, setValues + numValues );
        
        INDEX_TYPE const numUniqueValues = sortedArrayManipulation::removeDuplicates( setValues, numValues );
        arrayManipulation::resize( setValues, numValues, numUniqueValues );
        m_sizes[ i ] = numUniqueValues;
      }
    }

#ifdef ARRAY_BOUNDS_CHECK
    consistencyCheck();
#endif
  }

  void clearSet( INDEX_TYPE const i ) restrict_this
  { 
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    INDEX_TYPE const prevSize = sizeOfSet( i );
    T * const values = getSetValues( i );
    arrayManipulation::resize( values, prevSize, INDEX_TYPE( 0 ) );
    m_sizes[ i ] = 0;
  }

#ifdef USE_CHAI
  
  inline
  void move(chai::ExecutionSpace const space, bool const touch=true) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::move(space, touch); }

  inline
  void registerTouch(chai::ExecutionSpace const space) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::registerTouch(space); }
#endif

  inline
  void reserve( INDEX_TYPE const newCapacity ) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::reserve( newCapacity ); }

  inline
  void reserveValues( INDEX_TYPE const nnz ) restrict_this
  { m_values.reserve( nnz ); }

  inline
  void setCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::setCapacityOfArray(i, newCapacity); }

  inline
  void reserveCapacityOfSet( INDEX_TYPE const i, INDEX_TYPE newCapacity ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );
    if ( newCapacity > capacityOfSet( i ) ) setCapacityOfSet( i, newCapacity );
  }

  inline
  void compress() restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::compress(); }

  void resize( INDEX_TYPE const numSets ) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::resize( numSets, 0 ); }

  void resize( INDEX_TYPE const numSets, INDEX_TYPE const defaultSetCapacity ) restrict_this
  { ArrayOfSetsView<T, INDEX_TYPE>::resize( numSets, defaultSetCapacity ); }

  inline
  void appendSet( INDEX_TYPE const n=0 ) restrict_this
  {
    INDEX_TYPE const nSets = size();
    INDEX_TYPE const totalSize = m_offsets[nSets];

    m_offsets.push_back(totalSize);
    m_sizes.push_back(0);
    
    setCapacityOfSet( nSets, n );
  }

  inline
  void insertSet( INDEX_TYPE const i, INDEX_TYPE const n=0 ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_INSERT_BOUNDS( i );
    GEOS_ASSERT( arrayManipulation::isPositive( n ) );

    // Insert an set of capacity zero at the given location 
    INDEX_TYPE const offset = m_offsets[i];
    m_sizes.insert( i, 0 );
    m_offsets.insert( i + 1, offset );

    // Set the capacity of the new set
    setCapacityOfSet( i, n );
  }

  inline
  void eraseSet( INDEX_TYPE const i ) restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( i );

    setCapacityOfSet( i, 0 );
    m_sizes.erase( i );
    m_offsets.erase( i + 1 );
  }

  inline
  bool insertIntoSet( INDEX_TYPE const i, T const & val ) restrict_this
  { return ArrayOfSetsView<T, INDEX_TYPE>::insertIntoSetImpl( i, val, CallBacks( *this, i ) ); }

  inline
  INDEX_TYPE insertIntoSet( INDEX_TYPE const i, T const * const vals, INDEX_TYPE const n ) restrict_this
  { return ArrayOfSetsView<T, INDEX_TYPE>::insertIntoSetImpl( i, vals, n, CallBacks( *this, i ) ); }

  inline
  INDEX_TYPE insertSortedIntoSet( INDEX_TYPE const i, T const * const vals, INDEX_TYPE const n ) restrict_this
  { return ArrayOfSetsView<T, INDEX_TYPE>::insertSortedIntoSetImpl( i, vals, n, CallBacks( *this, i ) ); }

private:

  inline
  void dynamicallyGrowSet( INDEX_TYPE const i, INDEX_TYPE const newSize ) restrict_this
  { setCapacityOfSet( i, newSize * 2 ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks<T, INDEX_TYPE>
  {
public:

    inline
    CallBacks( ArrayOfSets<T, INDEX_TYPE> & sp, INDEX_TYPE const i ):
      m_aos( sp ),
      m_i( i )
    {}

    
    inline
    T * incrementSize( INDEX_TYPE const nToAdd ) const restrict_this
    {
      INDEX_TYPE const newNNZ = m_aos.sizeOfSet( m_i ) + nToAdd;
      if( newNNZ > m_aos.capacityOfSet( m_i ) )
      {
        m_aos.dynamicallyGrowSet( m_i, newNNZ );
      }

      return m_aos.getSetValues( m_i );
    }

private:
    ArrayOfSets<T, INDEX_TYPE> & m_aos;
    INDEX_TYPE const m_i;
  };

  // Aliasing protected members of ArrayOfSetsView.
  using ArrayOfSetsView<T, INDEX_TYPE>::m_offsets;
  using ArrayOfSetsView<T, INDEX_TYPE>::m_sizes;
  using ArrayOfSetsView<T, INDEX_TYPE>::m_values;
};

} /* namespace LvArray */
