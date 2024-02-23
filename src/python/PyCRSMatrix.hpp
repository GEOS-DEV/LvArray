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
 * @file
 */

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "numpyHelpers.hpp"
#include "../CRSMatrix.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"
#include "../output.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <string>
#include <memory>
#include <typeindex>

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 * @class PyCRSMatrixWrapperBase
 * @brief Provides a virtual Python wrapper around a CRSMatrix.
 */
class PyCRSMatrixWrapperBase
{
public:

  /**
   * @brief Default destructor.
   */
  virtual ~PyCRSMatrixWrapperBase() = default;

  /**
   * @brief Return the access level for the matrix.
   * @return The access level for the matrix.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the matrix.
   * @param accessLevel The access level.
   * @param memorySpace The memory space.
   */
  virtual void setAccessLevel( int const accessLevel, int const memorySpace ) = 0;

  /**
   * @brief Return a string representing the underlying CRSMatrix type.
   * @return a string representing the underlying CRSMatrix type.
   */
  virtual std::string repr() const = 0;

  /**
   * @brief Return the type of the columns.
   * @return The type of the columns.
   */
  virtual std::type_index columnType() const = 0;

  /**
   * @brief Return the type of the entries.
   * @return The type of the entries.
   */
  virtual std::type_index entryType() const = 0;

  /**
   * @brief Return the number of rows in the matrix.
   * @return The number of rows in the matrix.
   * @note This wraps CRSMatrix::numRows.
   */
  virtual long long numRows() const = 0;

  /**
   * @brief Return the number of columns in the matrix.
   * @return The number of columns in the matrix.
   * @note This wraps CRSMatrix::numColumns.
   */
  virtual long long numColumns() const = 0;

  /**
   * @brief Return true iff the matrix is compressed (no extra space between the rows).
   * @return True iff the matrix is compressed (no extra space between the rows).
   */
  virtual bool isCompressed() const = 0;

  /**
   * @brief Return a 1D NumPy array of the columns for @c row.
   * @param row The row to query.
   * @return A 1D NumPy array of the columns for @c row.
   * @note This wraps CRSMatrix::getColumns.
   */
  virtual PyObject * getColumns( long long const row ) const = 0;

  /**
   * @brief Return a 1D NumPy array of the entries for @c row.
   * @param row The row to query.
   * @return A 1D NumPy array of the entries for @c row.
   * @note This wraps CRSMatrix::getEntries.
   */
  virtual PyObject * getEntries( long long const row ) const = 0;

  /**
   * @brief Return 3 1D NumPy arrays of the entries, columns and offsets that comprise the matrix.
   * @return 3 1D NumPy arrays of the entries, columns and offsets that comprise the matrix.
   */
  virtual std::array< PyObject *, 3 > getEntriesColumnsAndOffsets() const = 0;

  /**
   * @brief Resize the matrix.
   * @param numRows The new number of rows in the matrix.
   * @param numCols The new number of columns in the matrix.
   * @param initialRowCapacity The non-zero capacity per-row.
   * @note This wraps CRSMatrix::resize.
   */
  virtual void resize( long long const numRows,
                       long long const numCols,
                       long long const initialRowCapacity ) = 0;

  /**
   * @brief Compress the matrix so that there is no extra space between the rows.
   * @note This wraps CRSMatrix::compress.
   */
  virtual void compress() = 0;

  /**
   * @brief Insert new non zero entries into a row.
   * @param row The row to insert into.
   * @param cols The columns to insert, must be of type @c columnType() and length @c numCols.
   * @param entries The entries to insert, must be of type @c entryType() and length @c numCols.
   * @param numCols The number of columns to insert.
   * @return The number of entries inserted.
   * @note This wraps CRSMatrix::insertNonZeros.
   */
  virtual long long insertNonZeros( long long const row,
                                    void const * const cols,
                                    void const * const entries,
                                    long long const numCols ) = 0;

  /**
   * @brief Remove non zero entries from a row.
   * @param row The row to remove from.
   * @param cols The columns to remove, must be of type @c columnType() and length @c numCols.
   * @param numCols The number of columns to insert.
   * @return The number of entries removed.
   * @note This wraps CRSMatrix::removeNonZeros.
   */
  virtual long long removeNonZeros( long long const row,
                                    void const * const cols,
                                    long long const numCols ) = 0;

  /**
   * @brief Add to existing entries in a row.
   * @param row The row to add to.
   * @param cols The columns to add to, must be of type @c columnType() and length @c numCols.
   * @param vals The values to add, must be of type @c entryType() and length @c numCols.
   * @param numCols The number of columns to add to.
   * @note This wraps CRSMatrix::addToRowBinarySearchUnsorted.
   */
  virtual void addToRow( long long const row,
                         void const * const cols,
                         void const * const vals,
                         long long const numCols ) const = 0;

protected:

  /**
   * @brief Construct an empty PyCRSMatrixWrapperBase.
   */
  PyCRSMatrixWrapperBase():
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /// access level for the matrix.
  int m_accessLevel;
};

/**
 * @class PyCRSMatrixWrapper
 * @brief Provides a concrete implementation of PyCRSMatrixWrapperBase.
 * @tparam T The type of the entries in the CRSMatrix.
 * @tparam COL_TYPE The type of the columns in the CRSMatrix.
 * @tparam INDEX_TYPE The index type of the CRSMatrix.
 * @tparam BUFFER_TYPE The buffer type of the CRSMatrix.
 * @note This holds a reference to the wrapped CRSMatrix, you must ensure that the reference remains valid
 *   for the lifetime of this object.
 */
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyCRSMatrixWrapper final : public PyCRSMatrixWrapperBase
{
public:

  /**
   * @brief Construct a new Python wrapper around @c matrix.
   * @param matrix The CRSMatrix to wrap.
   */
  PyCRSMatrixWrapper( CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & matrix ):
    PyCRSMatrixWrapperBase(),
    m_matrix( matrix )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyCRSMatrixWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index columnType() const final override
  { return typeid( COL_TYPE ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index entryType() const final override
  { return typeid( T ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long numRows() const final override
  { return m_matrix.numRows(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long numColumns() const final override
  { return m_matrix.numColumns(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isCompressed() const final override
  {
    #if defined(USE_OPENMP)
    using EXEC_POLICY = RAJA::omp_parallel_for_exec;
    using REDUCE_POLICY = RAJA::omp_reduce;
    #else
    using EXEC_POLICY = RAJA::loop_exec;
    using REDUCE_POLICY = RAJA::seq_reduce;
    #endif

    RAJA::ReduceBitAnd< REDUCE_POLICY, bool > compressed( true );

    RAJA::forall< EXEC_POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, m_matrix.numRows() ),
                                 [compressed, this] ( INDEX_TYPE const row )
    {
      compressed &= m_matrix.numNonZeros( row ) == m_matrix.nonZeroCapacity( row );
    }
                                 );

    return compressed.get();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * getColumns( long long const row ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const columns = m_matrix.getColumns( convertedRow );
    INDEX_TYPE const nnz = m_matrix.numNonZeros( convertedRow );
    constexpr INDEX_TYPE unitStride = 1;
    return createNumPyArray( columns, false, 1, &nnz, &unitStride );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * getEntries( long long const row ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    T * const entries = m_matrix.getEntries( convertedRow );
    INDEX_TYPE const nnz = m_matrix.numNonZeros( convertedRow );
    constexpr INDEX_TYPE unitStride = 1;
    return createNumPyArray( entries, getAccessLevel(), 1, &nnz, &unitStride );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::array< PyObject *, 3 > getEntriesColumnsAndOffsets() const final override
  {
    T * const entries = m_matrix.getEntries( 0 );
    COL_TYPE const * const columns = m_matrix.getColumns( 0 );
    INDEX_TYPE const * const offsets = m_matrix.getOffsets();

    INDEX_TYPE const numNonZeros = offsets[ m_matrix.numRows() ]; // size of columns and entries
    INDEX_TYPE offsetSize = m_matrix.numRows() + 1;
    constexpr INDEX_TYPE unitStride = 1;
    return { createNumPyArray( entries, getAccessLevel() >= static_cast< int >( LvArray::python::PyModify::MODIFIABLE ), 1, &numNonZeros, &unitStride ),
             createNumPyArray( columns, false, 1, &numNonZeros, &unitStride ),
             createNumPyArray( offsets, false, 1, &offsetSize, &unitStride ) };
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int const accessLevel, int const memorySpace ) final override
  {
    LVARRAY_UNUSED_VARIABLE( memorySpace );
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void resize( long long const numRows,
                       long long const numCols,
                       long long const initialRowCapacity ) final override
  {
    INDEX_TYPE const convertedNumRows = integerConversion< INDEX_TYPE >( numRows );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    INDEX_TYPE const convertedCapacity = integerConversion< INDEX_TYPE >( initialRowCapacity );
    m_matrix.resize( convertedNumRows, convertedNumCols, convertedCapacity );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void compress() final override
  { m_matrix.compress(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long insertNonZeros( long long const row,
                                    void const * const cols,
                                    void const * const entries,
                                    long long const numCols ) final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    T const * const castedEntries = reinterpret_cast< T const * >( entries );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return integerConversion< long long >( m_matrix.insertNonZeros( convertedRow, castedCols, castedEntries, convertedNumCols ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long removeNonZeros( long long const row,
                                    void const * const cols,
                                    long long const numCols ) final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return integerConversion< long long >( m_matrix.removeNonZeros( convertedRow, castedCols, convertedNumCols ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void addToRow( long long const row,
                         void const * const cols,
                         void const * const vals,
                         long long const numCols ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    T const * const castedVals = reinterpret_cast< T const * >( vals );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return m_matrix.template addToRowBinarySearchUnsorted< RAJA::seq_atomic >( convertedRow,
                                                                               castedCols,
                                                                               castedVals,
                                                                               convertedNumCols );
  }

private:
  /// The CRSMatrix to wrap.
  CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & m_matrix;
};

/**
 * @brief Create a Python object corresponding to @c matrix.
 * @param matrix The matrix to export to Python.
 * @return A Python object corresponding to @c matrix.
 */
PyObject * create( std::unique_ptr< internal::PyCRSMatrixWrapperBase > && matrix );

} // namespace internal

/**
 * @brief Create a Python object corresponding to @c matrix.
 * @tparam T The type of the entries in the CRSMatrix.
 * @tparam COL_TYPE The type of the columns in the CRSMatrix.
 * @tparam INDEX_TYPE The index type of the CRSMatrix.
 * @tparam BUFFER_TYPE The buffer type of the CRSMatrix.
 * @param matrix The CRSMatrix to export to Python.
 * @return A Python object corresponding to @c matrix.
 * @note The returned Python object holds a reference to @c matrix, you must ensure
 *   the reference remains valid throughout the lifetime of the object.
 */
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & matrix )
{
  auto tmp = std::make_unique< internal::PyCRSMatrixWrapper< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > >( matrix );
  return internal::create( std::move( tmp ) );
}

/**
 * @brief Return the Python type for the CRSMatrix.
 * @return The Python type for the CRSMatrix.
 */
PyTypeObject * getPyCRSMatrixType();

} // namespace python
} // namespace LvArray
