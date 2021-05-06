/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkHelpers.hpp"
#include "tensorOps.hpp"

// TPL includes
#include <benchmark/benchmark.h>

namespace LvArray
{
namespace benchmarking
{

using VALUE_TYPE = double;
using FLOAT = double;
constexpr unsigned long THREADS_PER_BLOCK = 32;

#define TIMING_LOOP( KERNEL ) \
  for( auto _ : this->m_state ) \
  { \
    LVARRAY_UNUSED_VARIABLE( _ ); \
    KERNEL; \
    benchmark::ClobberMemory(); \
  } \


template< int M, typename PERM_2D, typename PERM_3D, typename POLICY >
class Eigendecomposition
{
public:
  Eigendecomposition( ::benchmark::State & state ):
    m_state( state )
  {
    INDEX_TYPE const numMatrices = m_state.range( 0 );

    m_matrices.resize( numMatrices, tensorOps::SYM_SIZE< M > );
    m_matrices.setName( "matrices" );
    m_eigenvalues.resize( numMatrices, M );
    m_eigenvalues.setName( "eigenvalues" );
    m_eigenvectors.resize( numMatrices, M, M );
    m_eigenvectors.setName( "eigenvectors" );

    int iter = 0;
    initialize( m_matrices.toSlice(), iter );
  }

  ~Eigendecomposition()
  {
    m_state.counters[ "Eigenvalues calculated" ] = ::benchmark::Counter( m_eigenvalues.size(),
                                                                         benchmark::Counter::kIsIterationInvariantRate,
                                                                         benchmark::Counter::OneK::kIs1000 );
  }

  void eigenvalues() const
  {
    ArrayViewT< VALUE_TYPE const, PERM_2D > const matrices = m_matrices.toViewConst();
    ArrayViewT< FLOAT, PERM_2D > const eigenvalues = m_eigenvalues.toView();
    TIMING_LOOP( eigenvaluesKernel( matrices, eigenvalues ) );
  }

  void eigenvectors() const
  {
    ArrayViewT< VALUE_TYPE const, PERM_2D > const matrices = m_matrices.toViewConst();
    ArrayViewT< FLOAT, PERM_2D > const eigenvalues = m_eigenvalues.toView();
    ArrayViewT< FLOAT, PERM_3D > const eigenvectors = m_eigenvectors.toView();
    TIMING_LOOP( eigenvectorsKernel( matrices, eigenvalues, eigenvectors ) );
  }


  static void eigenvaluesKernel( ArrayViewT< VALUE_TYPE const, PERM_2D > const & matrices,
                                 ArrayViewT< FLOAT, PERM_2D > const & eigenvalues );

  static void eigenvectorsKernel( ArrayViewT< VALUE_TYPE const, PERM_2D > const & matrices,
                                  ArrayViewT< FLOAT, PERM_2D > const & eigenvalues,
                                  ArrayViewT< FLOAT, PERM_3D > const & eigenvectors );



protected:
  ::benchmark::State & m_state;

  ArrayT< VALUE_TYPE, PERM_2D > m_matrices;
  ArrayT< FLOAT, PERM_2D > m_eigenvalues;
  ArrayT< FLOAT, PERM_3D > m_eigenvectors;
};

} // namespace benchmarking
} // namespace LvArray
