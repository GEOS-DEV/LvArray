/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"
#include "MallocBuffer.hpp"

#if defined(USE_CHA)
  #include "ChaiBuffer.hpp"
#endif

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// Sphinx start after examples
TEST( CRSMatrix, examples )
{
  // Create a matrix with three rows and three columns.
  LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > matrix( 2, 3 );
  EXPECT_EQ( matrix.numRows(), 2 );
  EXPECT_EQ( matrix.numColumns(), 3 );

  // Insert two entries into the first row.
  int const row0Columns[ 2 ] = { 0, 2 };
  double const row0Values[ 2 ] = { 4, 3 };
  matrix.insertNonZeros( 0, row0Columns, row0Values, 2 );
  EXPECT_EQ( matrix.numNonZeros( 0 ), 2 );

  // Insert three entries into the second row.
  int const row1Columns[ 3 ] = { 0, 1, 2 };
  double const row1Values[ 3 ] = { 55, -1, 4 };
  matrix.insertNonZeros( 1, row1Columns, row1Values, 3 );
  EXPECT_EQ( matrix.numNonZeros( 1 ), 3 );

  // The entire matrix has five non zero entries.
  EXPECT_EQ( matrix.numNonZeros(), 5 );

  // Row 0 does not have an entry for column 1.
  EXPECT_TRUE( matrix.empty( 0, 1 ) );

  // Row 1 does have an entry for column 1.
  EXPECT_FALSE( matrix.empty( 1, 1 ) );

  LvArray::ArraySlice< int const, 1, 0, std::ptrdiff_t > columns = matrix.getColumns( 0 );
  LvArray::ArraySlice< double, 1, 0, std::ptrdiff_t > entries = matrix.getEntries( 0 );

  // Check the entries of the matrix.
  EXPECT_EQ( columns.size(), 2 );
  EXPECT_EQ( columns[ 0 ], row0Columns[ 0 ] );
  EXPECT_EQ( entries[ 0 ], row0Values[ 0 ] );

  EXPECT_EQ( columns[ 1 ], row0Columns[ 1 ] );
  entries[ 1 ] += 10;
  EXPECT_EQ( entries[ 1 ], 10 + row0Values[ 1 ] );
}
// Sphinx end before examples

// Sphinx start after compress
TEST( CRSMatrix, compress )
{
  // Create a matrix with two rows and four columns where each row has capacity 3.
  LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > matrix( 2, 4, 3 );

  // Insert two entries into the first row.
  int const row0Columns[ 2 ] = { 0, 2 };
  double const row0Values[ 2 ] = { 4, 3 };
  matrix.insertNonZeros( 0, row0Columns, row0Values, 2 );
  EXPECT_EQ( matrix.numNonZeros( 0 ), 2 );
  EXPECT_EQ( matrix.nonZeroCapacity( 0 ), 3 );

  // Insert two entries into the second row.
  int const row1Columns[ 3 ] = { 0, 1 };
  double const row1Values[ 3 ] = { 55, -1 };
  matrix.insertNonZeros( 1, row1Columns, row1Values, 2 );
  EXPECT_EQ( matrix.numNonZeros( 1 ), 2 );
  EXPECT_EQ( matrix.nonZeroCapacity( 1 ), 3 );

  // The rows are not adjacent in memory.
  EXPECT_NE( &matrix.getColumns( 0 )[ 0 ] + matrix.numNonZeros( 0 ), &matrix.getColumns( 1 )[ 0 ] );
  EXPECT_NE( &matrix.getEntries( 0 )[ 0 ] + matrix.numNonZeros( 0 ), &matrix.getEntries( 1 )[ 0 ] );

  // After compression the rows are adjacent in memroy.
  matrix.compress();
  EXPECT_EQ( &matrix.getColumns( 0 )[ 0 ] + matrix.numNonZeros( 0 ), &matrix.getColumns( 1 )[ 0 ] );
  EXPECT_EQ( &matrix.getEntries( 0 )[ 0 ] + matrix.numNonZeros( 0 ), &matrix.getEntries( 1 )[ 0 ] );
  EXPECT_EQ( matrix.numNonZeros( 0 ), matrix.nonZeroCapacity( 0 ) );
  EXPECT_EQ( matrix.numNonZeros( 1 ), matrix.nonZeroCapacity( 1 ) );

  // The entries in the matrix are unchanged.
  EXPECT_EQ( matrix.getColumns( 0 )[ 0 ], row0Columns[ 0 ] );
  EXPECT_EQ( matrix.getEntries( 0 )[ 0 ], row0Values[ 0 ] );
  EXPECT_EQ( matrix.getColumns( 0 )[ 1 ], row0Columns[ 1 ] );
  EXPECT_EQ( matrix.getEntries( 0 )[ 1 ], row0Values[ 1 ] );

  EXPECT_EQ( matrix.getColumns( 1 )[ 0 ], row1Columns[ 0 ] );
  EXPECT_EQ( matrix.getEntries( 1 )[ 0 ], row1Values[ 0 ] );
  EXPECT_EQ( matrix.getColumns( 1 )[ 1 ], row1Columns[ 1 ] );
  EXPECT_EQ( matrix.getEntries( 1 )[ 1 ], row1Values[ 1 ] );
}
// Sphinx end before compress


// Sphinx start after assimilate
TEST( CRSMatrix, assimilate )
{
  // Create a sparsity pattern with two rows and four columns where each row has capacity 3.
  LvArray::SparsityPattern< int, std::ptrdiff_t, LvArray::MallocBuffer > sparsity( 2, 4, 3 );

  // Insert two entries into the first row.
  int const row0Columns[ 2 ] = { 0, 2 };
  sparsity.insertNonZeros( 0, row0Columns, row0Columns + 2 );

  // Insert two entries into the second row.
  int const row1Columns[ 3 ] = { 0, 1 };
  sparsity.insertNonZeros( 1, row1Columns, row1Columns + 2 );

  // Create a matrix with the sparsity pattern
  LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > matrix;
  matrix.assimilate< RAJA::seq_exec >( std::move( sparsity ) );

  // The sparsity is empty after being assimilated.
  EXPECT_EQ( sparsity.numRows(), 0 );
  EXPECT_EQ( sparsity.numColumns(), 0 );

  // The matrix has the same shape as the sparsity pattern.
  EXPECT_EQ( matrix.numRows(), 2 );
  EXPECT_EQ( matrix.numColumns(), 4 );

  EXPECT_EQ( matrix.numNonZeros( 0 ), 2 );
  EXPECT_EQ( matrix.nonZeroCapacity( 0 ), 3 );

  EXPECT_EQ( matrix.numNonZeros( 1 ), 2 );
  EXPECT_EQ( matrix.nonZeroCapacity( 1 ), 3 );

  // The entries in the matrix are zero initialized.
  EXPECT_EQ( matrix.getColumns( 0 )[ 0 ], row0Columns[ 0 ] );
  EXPECT_EQ( matrix.getEntries( 0 )[ 0 ], 0 );
  EXPECT_EQ( matrix.getColumns( 0 )[ 1 ], row0Columns[ 1 ] );
  EXPECT_EQ( matrix.getEntries( 0 )[ 1 ], 0 );

  EXPECT_EQ( matrix.getColumns( 1 )[ 0 ], row1Columns[ 0 ] );
  EXPECT_EQ( matrix.getEntries( 1 )[ 0 ], 0 );
  EXPECT_EQ( matrix.getColumns( 1 )[ 1 ], row1Columns[ 1 ] );
  EXPECT_EQ( matrix.getEntries( 1 )[ 1 ], 0 );
}
// Sphinx end before assimilate

#if defined(RAJA_ENABLE_OPENMP)
// Sphinx start after views
TEST( CRSMatrix, views )
{
  // Create a 100x100 tri-diagonal sparsity pattern.
  LvArray::SparsityPattern< int, std::ptrdiff_t, LvArray::MallocBuffer > sparsity( 100, 100, 3 );

  // Since enough space has been preallocated in each row we can insert into the sparsity
  // pattern in parallel.
  LvArray::SparsityPatternView< int,
                                std::ptrdiff_t const,
                                LvArray::MallocBuffer > const sparsityView = sparsity.toView();

  RAJA::forall< RAJA::omp_parallel_for_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, sparsityView.numRows() ),
    [sparsityView] ( std::ptrdiff_t const row )
  {
    int const columns[ 3 ] = { int( row - 1 ), int( row ), int( row + 1 ) };
    int const begin = row == 0 ? 1 : 0;
    int const end = row == sparsityView.numRows() - 1 ? 2 : 3;
    sparsityView.insertNonZeros( row, columns + begin, columns + end );
  }
    );

  // Create a matrix from the sparsity pattern
  LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > matrix;
  matrix.assimilate< RAJA::omp_parallel_for_exec >( std::move( sparsity ) );

  // Assemble into the matrix.
  LvArray::CRSMatrixView< double,
                          int const,
                          std::ptrdiff_t const,
                          LvArray::MallocBuffer > const matrixView = matrix.toViewConstSizes();
  RAJA::forall< RAJA::omp_parallel_for_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, matrixView.numRows() ),
    [matrixView] ( std::ptrdiff_t const row )
  {
    // Some silly assembly where each diagonal entry ( r, r ) has a contribution to the row above ( r-1, r )
    // The current row (r, r-1), (r, r), (r, r+1) and the row below (r+1, r).
    int const columns[ 3 ] = { int( row - 1 ), int( row ), int( row + 1 ) };
    double const contribution[ 3 ] = { double( row ), double( row ), double( row ) };

    // Contribution to the row above
    if( row > 0 )
    {
      matrixView.addToRow< RAJA::builtin_atomic >( row - 1, columns + 1, contribution, 1 );
    }

    // Contribution to the current row
    int const begin = row == 0 ? 1 : 0;
    int const end = row == matrixView.numRows() - 1 ? 2 : 3;
    matrixView.addToRow< RAJA::builtin_atomic >( row, columns + begin, contribution, end - begin );

    // Contribution to the row below
    if( row < matrixView.numRows() - 1 )
    {
      matrixView.addToRow< RAJA::builtin_atomic >( row + 1, columns + 1, contribution, 1 );
    }
  }
    );

  // Check every row except for the first and the last.
  for( std::ptrdiff_t row = 1; row < matrix.numRows() - 1; ++row )
  {
    EXPECT_EQ( matrix.numNonZeros( row ), 3 );

    LvArray::ArraySlice< int const, 1, 0, std::ptrdiff_t > const rowColumns = matrix.getColumns( row );
    LvArray::ArraySlice< double const, 1, 0, std::ptrdiff_t > const rowEntries = matrix.getEntries( row );

    for( std::ptrdiff_t i = 0; i < matrix.numNonZeros( row ); ++i )
    {
      // The first first entry is the sum of the contributions of the row above and the current row.
      EXPECT_EQ( rowColumns[ 0 ], row - 1 );
      EXPECT_EQ( rowEntries[ 0 ], row + row - 1 );

      // The second entry is the from the current row alone.
      EXPECT_EQ( rowColumns[ 1 ], row );
      EXPECT_EQ( rowEntries[ 1 ], row );

      // The third entry is the sum of the contributions of the row below and the current row.
      EXPECT_EQ( rowColumns[ 2 ], row + 1 );
      EXPECT_EQ( rowEntries[ 2 ], row + row + 1 );
    }
  }

  // Check the first row.
  EXPECT_EQ( matrix.numNonZeros( 0 ), 2 );
  EXPECT_EQ( matrix.getColumns( 0 )[ 0 ], 0 );
  EXPECT_EQ( matrix.getEntries( 0 )[ 0 ], 0 );
  EXPECT_EQ( matrix.getColumns( 0 )[ 1 ], 1 );
  EXPECT_EQ( matrix.getEntries( 0 )[ 1 ], 1 );

  // Check the last row.
  EXPECT_EQ( matrix.numNonZeros( matrix.numRows() - 1 ), 2 );
  EXPECT_EQ( matrix.getColumns( matrix.numRows() - 1 )[ 0 ], matrix.numRows() - 2 );
  EXPECT_EQ( matrix.getEntries( matrix.numRows() - 1 )[ 0 ], matrix.numRows() - 1  + matrix.numRows() - 2 );
  EXPECT_EQ( matrix.getColumns( matrix.numRows() - 1 )[ 1 ], matrix.numRows() - 1 );
  EXPECT_EQ( matrix.getEntries( matrix.numRows() - 1 )[ 1 ], matrix.numRows() - 1 );
}
// Sphinx end before views
#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
