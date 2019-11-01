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

#include "CRSMatrix.hpp"
#include "testUtils.hpp"
#include "gtest/gtest.h"

#ifdef USE_CUDA

#include "Array.hpp"

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

#include <vector>
#include <set>
#include <utility>
#include <random>

namespace LvArray
{

// Alias for CRSMatrixView
template <class T, class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t>
using ViewType = CRSMatrixView<T, COL_TYPE, INDEX_TYPE const>;

using INDEX_TYPE = std::ptrdiff_t;
using COL_TYPE = unsigned int;

// A row is a sorted collection of unique columns each paired with a value, which is a set pairs.
template <class T>
using ROW_REF_TYPE = std::set<std::pair<T, COL_TYPE>, PairComp<T, COL_TYPE>>;

// A matrix is an array of rows.
template <class T>
using REF_TYPE = std::vector<ROW_REF_TYPE<T>>;

namespace internal
{

// A static random number generator is used so that subsequent calls to the same test method will
// do different things. This initializes the generator with the default seed so the test behavior will
// be uniform across runs and platforms. However the behavior of an individual method will depend on the state
// of the generator and hence on the calls before it.
static std::mt19937_64 gen;

/**
 * @brief Check that the CRSMatrixView is equivalent to the reference type.
 * @param [in] m the CRSMatrixView to check.
 * @param [in] mRef the reference to check against.
 */
template <class T>
void compareToReference(ViewType<T const, COL_TYPE const> const & m,
                        REF_TYPE<T> const & mRef)
{
  INDEX_TYPE const numRows = m.numRows();
  ASSERT_EQ(numRows, mRef.size());

  INDEX_TYPE ref_nnz = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    INDEX_TYPE const rowNNZ = m.numNonZeros(row);
    INDEX_TYPE const refRowNNZ = mRef[row].size();
    ref_nnz += refRowNNZ; 

    ASSERT_EQ(rowNNZ, refRowNNZ);

    if (rowNNZ == 0)
    {
      ASSERT_TRUE(m.empty(row));
    }
    else
    {
      ASSERT_FALSE(m.empty(row));
    }

    typename ROW_REF_TYPE<T>::iterator it = mRef[row].begin();
    COL_TYPE const * const columns = m.getColumns(row);
    T const * const entries = m.getEntries(row);
    for (INDEX_TYPE i = 0; i < rowNNZ; ++i)
    {
      EXPECT_FALSE(m.empty(row, columns[i]));
      EXPECT_EQ(entries[i], it->first);
      EXPECT_EQ(columns[i], it->second);
      ++it;
    }
  }

  ASSERT_EQ(m.numNonZeros(), ref_nnz);

  if (m.numNonZeros() == 0)
  {
    ASSERT_TRUE(m.empty());
  }
}

/**
 * @brief Test the insert method of the CRSMatrix.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_INSERTS the number of times to call insert.
 */
template <class T>
void insertTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef, INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & row_ref = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), row_ref.size());

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      COL_TYPE const col = columnDist(gen);
      T const value = T(col);
      ASSERT_EQ(row_ref.insert(std::pair<T, COL_TYPE>(value, col)).second, m.insertNonZero(row, col, value));
    }
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the insert multiple method of the CRSMatrix.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_INSERTS the number of entries to insert at a time.
 */
template <class T>
void insertMultipleTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef, INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::vector<COL_TYPE> columns(MAX_INSERTS);
  std::vector<T> entries(MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), refRow.size());

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columns[i] = columnDist(gen);
      entries[i] = T(columns[i]);
    }

    m.insertNonZeros(row, columns.data(), entries.data(), MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      refRow.insert(std::pair<T, COL_TYPE>(entries[i], columns[i]));
    }

    ASSERT_EQ(m.numNonZeros(row), refRow.size());
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the insert sorted method of the CRSMatrix.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_INSERTS the number of entries to insert at a time.
 */
template <class T>
void insertSortedTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef, INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::vector<COL_TYPE> columns(MAX_INSERTS);
  std::vector<T> entries(MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), refRow.size());

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columns[i] = columnDist(gen);
    }

    std::sort(columns.begin(), columns.begin() + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      entries[i] = T(columns[i]);
    }

    m.insertNonZerosSorted(row, columns.data(), entries.data(), MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      refRow.insert(std::pair<T, COL_TYPE>(entries[i], columns[i]));
    }

    ASSERT_EQ(m.numNonZeros(row), refRow.size());
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the remove method of the CRSMatrixView.
 * @param [in/out] m the CRSMatrixView to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_REMOVES the number of times to call remove.
 */
template <class T>
void removeTest(ViewType<T> const & m, REF_TYPE<T> & mRef, INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), refRow.size());
    
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      COL_TYPE const col = columnDist(gen);
      bool result = m.removeNonZero(row, col);
      ASSERT_EQ(result, refRow.erase(std::pair<T, COL_TYPE>(T(col), col)));
    }
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the remove multiple method of the CRSMatrixView.
 * @param [in/out] m the CRSMatrixView to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_REMOVES the number of entries to remove at a time.
 */
template <class T>
void removeMultipleTest(ViewType<T> const & m, REF_TYPE<T> & mRef,
                        INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::vector<COL_TYPE> columns(MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), refRow.size());

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columns[i] = columnDist(gen);
    }

    m.removeNonZeros(row, columns.data(), MAX_REMOVES);

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      refRow.erase(std::pair<T, COL_TYPE>(T(columns[i]), columns[i]));
    }

    ASSERT_EQ(m.numNonZeros(row), refRow.size());
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the remove sorted method of the CRSMatrixView.
 * @param [in/out] m the CRSMatrixView to test.
 * @param [in/out] mRef the reference to check against.
 * @param [in] MAX_REMOVES the number of entries to remove at a time.
 */
template <class T>
void removeSortedTest(ViewType<T> const & m, REF_TYPE<T> & mRef,
                      INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();
  ASSERT_EQ(numRows, mRef.size());

  std::vector<COL_TYPE> columns(MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    ASSERT_EQ(m.numNonZeros(row), refRow.size());

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columns[i] = columnDist(gen);
    }

    std::sort(columns.begin(), columns.begin() + MAX_REMOVES);

    m.removeNonZerosSorted(row, columns.data(), MAX_REMOVES);

    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      refRow.erase(std::pair<T, COL_TYPE>(T(columns[i]), columns[i]));
    }

    ASSERT_EQ(m.numNonZeros(row), refRow.size());
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the empty method of the CRSMatrixView.
 * @param [in] m the CRSMatrixView to test.
 * @param [in] mRef the reference to check against.
 */
template <class T>
void emptyTest(ViewType<T const, COL_TYPE const> const & m, REF_TYPE<T> const & mRef)
{
  INDEX_TYPE const numRows = m.numRows();
  ASSERT_EQ(numRows, mRef.size());

  SparsityPatternView<COL_TYPE const, INDEX_TYPE const> const & spView = m.toSparsityPatternView();

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> const & refRow = mRef[row];
    INDEX_TYPE const nnz = m.numNonZeros(row);
    ASSERT_EQ(nnz, refRow.size());

    typename ROW_REF_TYPE<T>::iterator it = refRow.begin();
    for (INDEX_TYPE i = 0; i < nnz; ++i)
    {
      COL_TYPE const col = it->second;
      EXPECT_FALSE(m.empty(row, col));
      ASSERT_EQ(m.empty(row, col), spView.empty(row, col));

      EXPECT_NE(m.empty(row, COL_TYPE(i)), refRow.count(std::pair<T, COL_TYPE>(T(), COL_TYPE(i))));
      ASSERT_EQ(m.empty(row, COL_TYPE(i)), spView.empty(row, COL_TYPE(i)));
      ++it;
    }
  }
}

/**
 * @brief Fill a row of the CRSMatrixView up to its capacity.
 * @param [in/out] m the CRSMatrixView.
 * @param [in/out] refRow the reference row to compare against.
 * @param [in] row the row of the CRSMatrixView to fill.
 */
template <class T>
void fillRow(ViewType<T> const & m, ROW_REF_TYPE<T> & refRow, INDEX_TYPE const row)
{
  INDEX_TYPE nToInsert = m.nonZeroCapacity(row) - m.numNonZeros(row);
  COL_TYPE testColumn = COL_TYPE(m.numColumns() - 1);
  COL_TYPE const * const columns = m.getColumns(row);
  T const * const entries = m.getEntries(row);

  ASSERT_GE(m.numColumns(), m.nonZeroCapacity(row));

  while (nToInsert > 0)
  {
    T const value(testColumn);
    bool const success = refRow.insert(std::pair<T, COL_TYPE>(value, testColumn)).second;
    ASSERT_EQ(success, m.insertNonZero(row, testColumn, value));
    nToInsert -= success;
    testColumn -= 1;
  }

  ASSERT_EQ(m.numNonZeros(row), refRow.size());
  ASSERT_EQ(m.nonZeroCapacity(row), m.numNonZeros(row));

  COL_TYPE const * const newColumns = m.getColumns(row);
  T const * const newEntries = m.getEntries(row);
  ASSERT_EQ(columns, newColumns);
  ASSERT_EQ(entries, newEntries);
}

/**
 * @brief Test the setRowCapacity method of CRSMatrix.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare against.
 */
template <class T>
void rowCapacityTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef)
{
  INDEX_TYPE const numRows = m.numRows();
  ASSERT_EQ(numRows, mRef.size());

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> & refRow = mRef[row];
    INDEX_TYPE const rowNNZ = m.numNonZeros(row);
    ASSERT_EQ(rowNNZ, refRow.size());

    // The new capacity will be the index of the row.
    INDEX_TYPE const new_capacity = row;
    if (new_capacity < rowNNZ)
    {
      COL_TYPE const * const columns = m.getColumns(row);
      T const * const entries = m.getEntries(row);
      m.setRowCapacity(row, new_capacity);

      ASSERT_EQ(m.nonZeroCapacity(row), new_capacity);
      ASSERT_EQ(m.nonZeroCapacity(row), m.numNonZeros(row));

      // Check that the pointers to the columns and entries haven't changed.
      COL_TYPE const * const newColumns = m.getColumns(row);
      T const * const newEntries = m.getEntries(row);
      ASSERT_EQ(columns, newColumns);
      ASSERT_EQ(entries, newEntries);

      // Erase the last entries from the reference.
      INDEX_TYPE const difference = rowNNZ - new_capacity;
      typename ROW_REF_TYPE<T>::iterator erase_pos = refRow.end();
      std::advance(erase_pos, -difference);
      refRow.erase(erase_pos, refRow.end());
      
      ASSERT_EQ(m.numNonZeros(row), refRow.size());
    }
    else
    {
      m.setRowCapacity(row, new_capacity);
      ASSERT_EQ(m.nonZeroCapacity(row), new_capacity);
      ASSERT_EQ(m.numNonZeros(row), rowNNZ);
      
      fillRow(m.toView(), refRow, row);      
    }
  }

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the copy constructor of the CRSMatrix.
 * @param [in] m the CRSMatrix to check.
 * @param [in] mRef the reference to check against.
 */
template <class T>
void deepCopyTest(CRSMatrix<T> const & m, REF_TYPE<T> const & mRef)
{
  CRSMatrix<T> copy(m);

  ASSERT_EQ(m.numRows(), copy.numRows());
  ASSERT_EQ(m.numColumns(), copy.numColumns());
  ASSERT_EQ(m.numNonZeros(), copy.numNonZeros());

  INDEX_TYPE const totalNNZ = m.numNonZeros();

  for (INDEX_TYPE row = 0; row < m.numRows(); ++row)
  {
    INDEX_TYPE const nnz = m.numNonZeros(row);
    ASSERT_EQ(nnz, copy.numNonZeros(row));

    COL_TYPE const * const cols = m.getColumns(row);
    COL_TYPE const * const cols_cpy = copy.getColumns(row);
    ASSERT_NE(cols, cols_cpy);

    T const * const entries = m.getEntries(row); 
    T const * const entries_cpy = copy.getEntries(row); 
    ASSERT_NE(entries, entries_cpy);

    // Iterate backwards and remove entries from copy.
    for (INDEX_TYPE i = nnz - 1; i >= 0; --i)
    {
      EXPECT_EQ(cols[i], cols_cpy[i]);
      EXPECT_EQ(entries[i], entries_cpy[i]);

      copy.removeNonZero(row, cols_cpy[i]);
    }

    EXPECT_EQ(copy.numNonZeros(row), 0);
    EXPECT_EQ(m.numNonZeros(row), nnz);
  }

  EXPECT_EQ(copy.numNonZeros(), 0);
  EXPECT_EQ(m.numNonZeros(), totalNNZ);

  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the copy constructor of the CRSMatrixView.
 * @param [in/out] m the CRSMatrix to check.
 */
template <class T>
void shallowCopyTest(CRSMatrix<T> const & m)
{
  ViewType<T> copy(m);

  ASSERT_EQ(m.numRows(), copy.numRows());
  ASSERT_EQ(m.numColumns(), copy.numColumns());
  ASSERT_EQ(m.numNonZeros(), copy.numNonZeros());

  for (INDEX_TYPE row = 0; row < m.numRows(); ++row)
  {
    INDEX_TYPE const nnz = m.numNonZeros(row);
    ASSERT_EQ(nnz, copy.numNonZeros(row));

    COL_TYPE const * const cols = m.getColumns(row);
    COL_TYPE const * const cols_cpy = copy.getColumns(row);
    ASSERT_EQ(cols, cols_cpy);

    T const * const entries = m.getEntries(row); 
    T const * const entries_cpy = copy.getEntries(row);
    ASSERT_EQ(entries, entries_cpy);

    // Iterate backwards and remove entries from copy.
    for (INDEX_TYPE i = nnz - 1; i >= 0; --i)
    {
      EXPECT_EQ(cols[i], cols_cpy[i]);
      EXPECT_EQ(entries[i], entries_cpy[i]);

      copy.removeNonZero(row, cols_cpy[i]);
    }

    EXPECT_EQ(copy.numNonZeros(row), 0);
    EXPECT_EQ(m.numNonZeros(row), 0);
  }

  EXPECT_EQ(copy.numNonZeros(), 0);
  EXPECT_EQ(m.numNonZeros(), 0);
  EXPECT_TRUE(copy.empty());
  EXPECT_TRUE(m.empty());
}

#ifdef USE_CUDA

/**
 * @brief Test the CRSMatrixView copy constructor in regards to memory motion.
 * @param [in/out] m the CRSMatrixView to test.
 */
template <class T>
void memoryMotionTest(ViewType<T> const & m)
{
  INDEX_TYPE const numRows = m.numRows();

  // Set the columns and entries. The const casts here isn't necessary, we could remove and insert
  // into the view instead, but this is quicker.
  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = m.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    T * const entries = m.getEntries(row);
    for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
    {
      columnsNC[i] = curIndex;
      entries[i] = T(curIndex++);
    }
  }

  // Capture the view on device and set the entries. Here the const cast is necessary
  // because we don't want to test the insert/remove methods on device yet.
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = m.getColumns(row);
      COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
      T * const entries = m.getEntries(row);
      for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
      {
        columnsNC[i] += columns[i];
        entries[i] += entries[i];
      }
    }
  );

  // Check that the columns and entries have been updated.
  curIndex = 0;
  forall(sequential(), 0, numRows,
    [=, &curIndex](INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
      {
        COL_TYPE col = COL_TYPE(curIndex);
        col += col;
        T val = T(curIndex++);
        val += val;
        EXPECT_EQ(col, m.getColumns(row)[i]);
        EXPECT_EQ(val, m.getEntries(row)[i]);
      }
    }
  );
}

/**
 * @brief Test the CRSMatrix move method.
 * @param [in/out] m the CRSMatrix to test.
 */
template <class T>
void memoryMotionMove(CRSMatrix<T> & m)
{
  INDEX_TYPE const numRows = m.numRows();

  ViewType<T> const & view = m;

  INDEX_TYPE curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = view.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    T * const entries = view.getEntries(row);
    for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
    {
      columnsNC[i] = curIndex;
      entries[i] = T(curIndex++);
    }
  }

  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = view.getColumns(row);
      COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
      T * const entries = view.getEntries(row);
      for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
      {
        columnsNC[i] += columns[i];
        entries[i] += entries[i];
      }
    }
  );

  m.move(chai::CPU);
  curIndex = 0;
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < view.numNonZeros(row); ++i)
    {
      COL_TYPE col = COL_TYPE(curIndex);
      col += col;
      T val = T(curIndex++);
      val += val;
      EXPECT_EQ(col, view.getColumns(row)[i]);
      EXPECT_EQ(val, view.getEntries(row)[i]);
    }
  }
}

/**
 * @brief Test the const capture semantics where COL_TYPE is const.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 */
template <class T>
void memoryMotionConstTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  // Create a view capture it on device and update the entries.
  ViewType<T, COL_TYPE const> const & mConst = m;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = mConst.getColumns(row);
      T * const entries = mConst.getEntries(row);
      for (INDEX_TYPE i = 0; i < mConst.numNonZeros(row); ++i)
      {
        GEOS_ERROR_IF(!arrayManipulation::isPositive(columns[i]) || columns[i] >= numCols, "Invalid column.");
        entries[i] = T(2 * i + 7);
      }
    }
  );

  // Fill in the rows on host.
  Array<INDEX_TYPE, 1> originalNNZ(numRows);
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    originalNNZ[row] = m.numNonZeros(row);
    fillRow(m.toView(), mRef[row], row);
  }

  compareToReference(m.toViewCC(), mRef);

  // This should copy back the entries but not the columns.
  m.move(chai::CPU);

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    ROW_REF_TYPE<T> const & row_ref = mRef[row];

    ASSERT_EQ(m.numNonZeros(row), row_ref.size());

    typename ROW_REF_TYPE<T>::iterator it = row_ref.begin();
    for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
    {

      // So the columns should be the same as the reference.
      COL_TYPE const col = m.getColumns(row)[i];
      EXPECT_EQ(col, it->second);

      // And the entries should be different.
      if (i < originalNNZ[row])
      {
        EXPECT_EQ(m.getEntries(row)[i], T(2 * i + 7));
      }

      ++it;
    }
  }
}

/**
 * @brief Test the const capture semantics where both T and COL_TYPE are const.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */

template <class T>
void memoryMotionConstConstTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef, INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  // Create a view and capture it on device.
  ViewType<T const, COL_TYPE const> const & mConst = m;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      COL_TYPE const * const columns = mConst.getColumns(row);
      T const * const entries = mConst.getEntries(row);
      for (INDEX_TYPE i = 0; i < mConst.numNonZeros(row); ++i)
      {
        GEOS_ERROR_IF(!arrayManipulation::isPositive(columns[i]) || columns[i] >= numCols, "Invalid column.");
        GEOS_ERROR_IF(entries[i] != T(columns[i]), "Incorrect value.");
      }
    }
  );

  // Insert entries, this will modify the offsets, sizes, columns and entries.
  insertTest(m, mRef, MAX_INSERTS);

  // Move the matrix back to the host, this should copy nothing.
  m.move(chai::CPU);

  // And therefore the matrix should still equal the reference.
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView insert method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class T>
void insertDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                      INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  // Create an Array of the columns to insert into each row.
  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  // Reserve space in each row for the number of insertions and populate the columnsToInsert Array.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    m.reserveNonZeros(row, m.numNonZeros() + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
    }
  }

  compareToReference(m.toViewCC(), mRef);

  // Create views and insert the columns on the device.
  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & insertView = columnsToInsert;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
      {
        COL_TYPE const col = insertView(row, i);
        mView.insertNonZero(row, col, T(col));
      }
    }
  );

  // Insert the entries into the reference.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      COL_TYPE const col = insertView(row, i);
      mRef[row].insert(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  // Move the matrix back to the host and compare against the reference.
  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView insert multiple method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class T>
void insertMultipleDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                              INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);
  Array<T, 2> entriesToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    m.reserveNonZeros(row, m.numNonZeros() + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
      entriesToInsert(row, i) = T(columnsToInsert(row, i));
    }
  }

  compareToReference(m.toViewCC(), mRef);

  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & colInsertView = columnsToInsert;
  ArrayView<T const, 2> const & valInsertView = entriesToInsert;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      mView.insertNonZeros(row, colInsertView[row], valInsertView[row], MAX_INSERTS);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      COL_TYPE const col = colInsertView(row, i);
      mRef[row].insert(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView insert sorted method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_INSERTS the maximum number of inserts.
 */
template <class T>
void insertSortedDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                            INDEX_TYPE const MAX_INSERTS)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  Array<COL_TYPE, 2> columnsToInsert(numRows, MAX_INSERTS);
  Array<T, 2> entriesToInsert(numRows, MAX_INSERTS);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    m.reserveNonZeros(row, m.numNonZeros() + MAX_INSERTS);

    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      columnsToInsert(row, i) = columnDist(gen);
    }

    // Sort the columns so they're ready to be inserted.
    COL_TYPE * const colPtr = columnsToInsert[row];
    std::sort(colPtr, colPtr + MAX_INSERTS);

    // Populate the entries to insert array.
    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      entriesToInsert(row, i) = T(columnsToInsert(row, i));
    }
  }


  compareToReference(m.toViewCC(), mRef);

  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & colInsertView = columnsToInsert;
  ArrayView<T const, 2> const & valInsertView = entriesToInsert;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      mView.insertNonZerosSorted(row, colInsertView[row], valInsertView[row], MAX_INSERTS);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_INSERTS; ++i)
    {
      COL_TYPE const col = colInsertView(row, i);
      mRef[row].insert(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView remove method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class T>
void removeDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                      INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  // Create an array of the columns to remove from each row.
  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  // Initialize the columns to remove.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }
  }

  // Create views and remove the columns on device.
  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & removeView = columnsToRemove;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
      {
        mView.removeNonZero(row, removeView(row, i));
      }
    }
  );

  // Remove the columns from mRef
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      COL_TYPE const col = removeView(row, i);
      mRef[row].erase(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  // Move the matrix back to the host and compare.
  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView remove multiple method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class T>
void removeMultipleDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                              INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }
  }

  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & colRemoveView = columnsToRemove;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      mView.removeNonZeros(row, colRemoveView[row], MAX_REMOVES);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      COL_TYPE const col = colRemoveView(row, i);
      mRef[row].erase(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the CRSMatrixView remove sorted method on device.
 * @param [in/out] m the CRSMatrix to test.
 * @param [in/out] mRef the reference to compare to.
 * @param [in] MAX_REMOVES the maximum number of removes.
 */
template <class T>
void removeSortedDeviceTest(CRSMatrix<T> & m, REF_TYPE<T> & mRef,
                            INDEX_TYPE const MAX_REMOVES)
{
  INDEX_TYPE const numRows = m.numRows();
  INDEX_TYPE const numCols = m.numColumns();

  Array<COL_TYPE, 2> columnsToRemove(numRows, MAX_REMOVES);

  std::uniform_int_distribution<COL_TYPE> columnDist(0, COL_TYPE(numCols - 1));

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      columnsToRemove(row, i) = columnDist(gen);
    }

    // Sort the columns so they're ready to be removed.
    COL_TYPE * const colPtr = columnsToRemove[row];
    std::sort(colPtr, colPtr + MAX_REMOVES);
  }

  ViewType<T> const & mView = m;
  ArrayView<COL_TYPE const, 2> const & colRemoveView = columnsToRemove;
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      mView.removeNonZerosSorted(row, colRemoveView[row], MAX_REMOVES);
    }
  );

  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    for (INDEX_TYPE i = 0; i < MAX_REMOVES; ++i)
    {
      COL_TYPE const col = colRemoveView(row, i);
      mRef[row].erase(std::pair<T, COL_TYPE>(T(col), col));
    }
  }

  m.move(chai::CPU);
  compareToReference(m.toViewCC(), mRef);
}

/**
 * @brief Test the empty method of the CRSMatrixView on device.
 * @param [in] v the CRSMatrixView to test.
 */
template <class T>
void emptyDeviceTest(ViewType<T const, COL_TYPE const> const & m)
{
  INDEX_TYPE const numRows = m.numRows();

  // Initialize each row to only contain even numbered columns.
  for (INDEX_TYPE row = 0; row < numRows; ++row)
  {
    COL_TYPE const * const columns = m.getColumns(row);
    COL_TYPE * const columnsNC = const_cast<COL_TYPE *>(columns);
    for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
    {
      columnsNC[i] = COL_TYPE(2 * i);
    }
  }

  // Check that each row contains the even columns and no odd columns on device.
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      for (INDEX_TYPE i = 0; i < m.numNonZeros(row); ++i)
      {
        GEOS_ERROR_IF(m.empty(row, COL_TYPE(2 * i)), "Row should contain even columns.");
        GEOS_ERROR_IF(!m.empty(row, COL_TYPE(2 * i + 1)), "Row should not contain odd columns.");       
      }
    }
  );
}

/**
 * @brief Test the toSparsityPatternView method of the CRSMatrixView and device capture.
 * @param [in] v the CRSMatrixView to test.
 */
template <class T>
void sparsityPatternViewDeviceTest(ViewType<T const, COL_TYPE const> const & m)
{
  INDEX_TYPE const numRows = m.numRows();

  SparsityPatternView<COL_TYPE const, INDEX_TYPE const> const & spView = m.toSparsityPatternView();

  // Check that each row contains the even columns and no odd columns on device.
  forall(cuda(), 0, numRows,
    [=] __device__ (INDEX_TYPE row)
    {
      INDEX_TYPE const nnz = m.numNonZeros(row);
      GEOS_ERROR_IF(nnz != spView.numNonZeros(row), "The number of non zeros should be equal!");

      COL_TYPE const * const columns = m.getColumns(row);
      COL_TYPE const * const spColumns = spView.getColumns(row);
      GEOS_ERROR_IF(columns != spColumns, "The column pointers should be equal!");

      for (INDEX_TYPE i = 0; i < nnz; ++i)
      {
        GEOS_ERROR_IF(columns[i] != spColumns[i], "Columns should be equal!");
        GEOS_ERROR_IF(m.empty(row, columns[i]) != spView.empty(row, columns[i]), "Should be equal.");
        GEOS_ERROR_IF(m.empty(row, COL_TYPE(i)) != spView.empty(row, COL_TYPE(i)), "Should be equal.");
      }
    }
  );

  SUCCEED();
}

#endif

} /* namespace internal */

TEST(CRSMatrix, construction)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE SIZE_HINT = 5;

  // Test the construction with no hint
  {
    CRSMatrix<double> m(NROWS, NCOLS);
    EXPECT_EQ(m.numRows(), NROWS);
    EXPECT_EQ(m.numColumns(), NCOLS);
    EXPECT_EQ(m.numNonZeros(), 0);
    EXPECT_TRUE(m.empty());

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      EXPECT_EQ(m.numNonZeros(row), 0);
      EXPECT_EQ(m.nonZeroCapacity(row), 0);
      EXPECT_EQ(m.empty(row), true);

      COL_TYPE const * const columns = m.getColumns(row);
      EXPECT_EQ(columns, nullptr);

      double const * const entries = m.getEntries(row);
      EXPECT_EQ(entries, nullptr);
    }
  }

  // Test the construction with a hint
  {
    CRSMatrix<double> m(NROWS, NCOLS, SIZE_HINT);
    EXPECT_EQ(m.numRows(), NROWS);
    EXPECT_EQ(m.numColumns(), NCOLS);
    EXPECT_EQ(m.numNonZeros(), 0);
    EXPECT_EQ(m.nonZeroCapacity(), NROWS * SIZE_HINT);
    EXPECT_TRUE(m.empty());

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      EXPECT_EQ(m.numNonZeros(row), 0);
      EXPECT_EQ(m.nonZeroCapacity(row), SIZE_HINT);
      EXPECT_EQ(m.empty(row), true);

      COL_TYPE const * const columns = m.getColumns(row);
      ASSERT_NE(columns, nullptr);

      double const * const entries = m.getEntries(row);
      ASSERT_NE(entries, nullptr);
    }
  }
}

TEST(CRSMatrix, insert)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);

    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);

    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);

    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::insertTest(m, ref, MAX_INSERTS);
  }
}

TEST(CRSMatrix, insertMultiple)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);

    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);

    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);

    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
    internal::insertMultipleTest(m, ref, MAX_INSERTS);
  }
}

TEST(CRSMatrix, insertSorted)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);

    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);

    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);

    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
    internal::insertSortedTest(m, ref, MAX_INSERTS);
  }
}

TEST(CRSMatrix, remove)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeTest(m.toView(), ref, MAX_REMOVES);
  }
}

TEST(CRSMatrix, removeMultiple)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeMultipleTest(m.toView(), ref, MAX_REMOVES);
  }
}

TEST(CRSMatrix, removeSorted)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::removeSortedTest(m.toView(), ref, MAX_REMOVES);
  }
}

TEST(CRSMatrix, empty)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr int NCOLS = 150;
  constexpr int MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::emptyTest(m.toViewCC(), ref);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::emptyTest(m.toViewCC(), ref);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::emptyTest(m.toViewCC(), ref);
  }
}

TEST(CRSMatrix, capacity)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);

    ASSERT_EQ(m.nonZeroCapacity(), 0);

    m.reserveNonZeros(NROWS * MAX_INSERTS);
    ASSERT_EQ(m.nonZeroCapacity(), NROWS * MAX_INSERTS);
    ASSERT_EQ(m.numNonZeros(), 0);

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      ASSERT_EQ(m.nonZeroCapacity(row), 0);
      ASSERT_EQ(m.numNonZeros(row), 0);
    }

    COL_TYPE const * const columns = m.getColumns(0);
    int const * const entries = m.getEntries(0);

    internal::insertTest(m, ref, MAX_INSERTS);

    COL_TYPE const * const newColumns = m.getColumns(0);
    int const * const newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);

    ASSERT_EQ(m.nonZeroCapacity(), 0);

    m.reserveNonZeros(NROWS * MAX_INSERTS);
    ASSERT_EQ(m.nonZeroCapacity(), NROWS * MAX_INSERTS);
    ASSERT_EQ(m.numNonZeros(), 0);

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      ASSERT_EQ(m.nonZeroCapacity(row), 0);
      ASSERT_EQ(m.numNonZeros(row), 0);
    }

    COL_TYPE const * const columns = m.getColumns(0);
    Tensor const * const entries = m.getEntries(0);
    
    internal::insertTest(m, ref, MAX_INSERTS);

    COL_TYPE const * const newColumns = m.getColumns(0);
    Tensor const * const newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);

    ASSERT_EQ(m.nonZeroCapacity(), 0);

    m.reserveNonZeros(NROWS * MAX_INSERTS);
    ASSERT_EQ(m.nonZeroCapacity(), NROWS * MAX_INSERTS);
    ASSERT_EQ(m.numNonZeros(), 0);

    for (INDEX_TYPE row = 0; row < NROWS; ++row)
    {
      ASSERT_EQ(m.nonZeroCapacity(row), 0);
      ASSERT_EQ(m.numNonZeros(row), 0);
    }

    COL_TYPE const * const columns = m.getColumns(0);
    TestString const * const entries = m.getEntries(0);
    
    internal::insertTest(m, ref, MAX_INSERTS);

    COL_TYPE const * const newColumns = m.getColumns(0);
    TestString const * const newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);
  }
}

TEST(CRSMatrix, rowCapacity)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<int> ref(NROWS);

    COL_TYPE const * const columns = m.getColumns(0);
    int const * const entries = m.getEntries(0);

    internal::insertTest(m, ref, MAX_INSERTS);

    COL_TYPE const * newColumns = m.getColumns(0);
    int const * newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);

    internal::rowCapacityTest(m, ref);

    newColumns = m.getColumns(0);
    newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<Tensor> ref(NROWS);

    COL_TYPE const * const columns = m.getColumns(0);
    Tensor const * const entries = m.getEntries(0);
    
    internal::insertTest(m, ref, MAX_INSERTS);
    
    COL_TYPE const * newColumns = m.getColumns(0);
    Tensor const * newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);

    internal::rowCapacityTest(m, ref);

    newColumns = m.getColumns(0);
    newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS, MAX_INSERTS);
    REF_TYPE<TestString> ref(NROWS);

    COL_TYPE const * const columns = m.getColumns(0);
    TestString const * const entries = m.getEntries(0);
    
    internal::insertTest(m, ref, MAX_INSERTS);
    
    COL_TYPE const * newColumns = m.getColumns(0);
    TestString const * newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
    ASSERT_EQ(entries, newEntries);

    internal::rowCapacityTest(m, ref);

    newColumns = m.getColumns(0);
    newEntries = m.getEntries(0);
    ASSERT_EQ(columns, newColumns);
  }
}

TEST(CRSMatrix, deepCopy)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::deepCopyTest(m, ref);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::deepCopyTest(m, ref);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::deepCopyTest(m, ref);
  }
}

TEST(CRSMatrix, shallowCopy)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::shallowCopyTest(m);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::shallowCopyTest(m);
  }

  {
    CRSMatrix<TestString> m(NROWS, NCOLS);
    REF_TYPE<TestString> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::shallowCopyTest(m);
  }
}

#ifdef USE_CUDA

CUDA_TEST(CRSMatrix, memoryMotion)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionTest(m.toView());
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionTest(m.toView());
  }
}

CUDA_TEST(CRSMatrix, memoryMotionMove)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionMove(m);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionMove(m);
  }
}

CUDA_TEST(CRSMatrix, memoryMotionConst)
{
  constexpr INDEX_TYPE NROWS = 1;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 10;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionConstTest(m, ref);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionConstTest(m, ref);
  }
}

CUDA_TEST(CRSMatrix, memoryMotionConstConst)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionConstConstTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::memoryMotionConstConstTest(m, ref, MAX_INSERTS);
  }
}

CUDA_TEST(CRSMatrix, insertDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
  }
}

CUDA_TEST(CRSMatrix, insertMultipleDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
    internal::insertMultipleDeviceTest(m, ref, MAX_INSERTS);
  }
}

CUDA_TEST(CRSMatrix, insertSortedDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
    internal::insertSortedDeviceTest(m, ref, MAX_INSERTS);
  }
}

CUDA_TEST(CRSMatrix, removeDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeDeviceTest(m, ref, MAX_REMOVES);
  }
}

CUDA_TEST(CRSMatrix, removeMultipleDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeMultipleDeviceTest(m, ref, MAX_REMOVES);
  }
}

CUDA_TEST(CRSMatrix, removeSortedDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr INDEX_TYPE NCOLS = 150;
  constexpr INDEX_TYPE MAX_INSERTS = 75;
  constexpr INDEX_TYPE MAX_REMOVES = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(m, ref, MAX_INSERTS);
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(m, ref, MAX_REMOVES);
    internal::insertDeviceTest(m, ref, MAX_INSERTS);
    internal::removeSortedDeviceTest(m, ref, MAX_REMOVES);
  }
}

TEST(CRSMatrix, emptyDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr int NCOLS = 150;
  constexpr int MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::emptyDeviceTest(m.toViewCC());
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::emptyDeviceTest(m.toViewCC());
  }
}

TEST(CRSMatrix, sparsityPatternViewDevice)
{
  constexpr INDEX_TYPE NROWS = 100;
  constexpr int NCOLS = 150;
  constexpr int MAX_INSERTS = 75;

  {
    CRSMatrix<int> m(NROWS, NCOLS);
    REF_TYPE<int> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::sparsityPatternViewDeviceTest(m.toViewCC());
  }

  {
    CRSMatrix<Tensor> m(NROWS, NCOLS);
    REF_TYPE<Tensor> ref(NROWS);
    internal::insertTest(m, ref, MAX_INSERTS);
    internal::sparsityPatternViewDeviceTest(m.toViewCC());
  }
}

#endif

} // namespace LvArray

int main( int argc, char* argv[] )
{
#if defined(USE_MPI)
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );
#else
  logger::InitializeLogger( );
#endif

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();

#ifdef USE_CHAI
  chai::ArrayManager::finalize();
#endif

#if defined(USE_MPI)
  MPI_Finalize();
#endif
  return result;
}
