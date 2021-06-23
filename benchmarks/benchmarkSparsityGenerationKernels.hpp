/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkHelpers.hpp"
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"
#include "ArrayOfArrays.hpp"
#include "system.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

using COLUMN_TYPE = std::ptrdiff_t;
using ENTRY_TYPE = double;
constexpr unsigned long THREADS_PER_BLOCK = 256;

#if defined(LVARRAY_USE_CUDA)
using ELEM_TO_NODE_PERM = RAJA::PERM_JI;
#else
using ELEM_TO_NODE_PERM = RAJA::PERM_JI;
#endif

using SparsityPatternT = SparsityPattern< COLUMN_TYPE, INDEX_TYPE, DEFAULT_BUFFER >;

using SparsityPatternViewT = SparsityPatternView< COLUMN_TYPE, INDEX_TYPE const, DEFAULT_BUFFER >;

using CRSMatrixT = CRSMatrix< ENTRY_TYPE, COLUMN_TYPE, INDEX_TYPE, DEFAULT_BUFFER >;

using CRSMatrixViewConstSizesT = CRSMatrixView< ENTRY_TYPE, COLUMN_TYPE const, INDEX_TYPE const, DEFAULT_BUFFER >;

constexpr int NDIM = 3;
constexpr int NODES_PER_ELEM = 8;
constexpr int MAX_ELEMS_PER_NODE = 8;
constexpr int MAX_COLUMNS_PER_ROW = 81;

class SparsityGenerationNative
{
public:

  SparsityGenerationNative( ::benchmark::State & state, bool const destructorCheck=true ):
    m_state( state ),
    m_numElemsX( state.range( 0 ) ),
    m_numElemsY( state.range( 1 ) ),
    m_numElemsZ( state.range( 2 ) ),
    m_numElems( m_numElemsX * m_numElemsY * m_numElemsZ ),
    m_numNodes( ( m_numElemsX + 1 ) * ( m_numElemsY + 1 ) * ( m_numElemsZ + 1 ) ),
    m_elemToNodeMap( m_numElems, NODES_PER_ELEM ),
    m_nodeToElemMap( m_numNodes, MAX_ELEMS_PER_NODE ),
    m_destructorCheck( destructorCheck )
  {
    CALI_CXX_MARK_SCOPE( "SparsityGenerationNative constructor" );

    INDEX_TYPE const elemJp = m_numElemsX;
    INDEX_TYPE const elemKp = elemJp * m_numElemsY;

    INDEX_TYPE const nodeJp = m_numElemsX + 1;
    INDEX_TYPE const nodeKp = nodeJp * ( m_numElemsY + 1 );

    constructStructuredElementToNodeMap( m_elemToNodeMap, m_numElemsX, m_numElemsY, m_numElemsZ );

    // Populate the node to element map.
    for( INDEX_TYPE i = 0; i < m_numElemsX + 1; ++i )
    {
      for( INDEX_TYPE j = 0; j < m_numElemsY + 1; ++j )
      {
        for( INDEX_TYPE k = 0; k < m_numElemsZ + 1; ++k )
        {
          INDEX_TYPE const nodeIndex = i + nodeJp * j + nodeKp * k;

          for( INDEX_TYPE di = -1; di < 1; ++di )
          {
            if( i + di < 0 || i + di >= m_numElemsX ) continue;
            for( INDEX_TYPE dj = -1; dj < 1; ++dj )
            {
              if( j + dj < 0 || j + dj >= m_numElemsY ) continue;
              for( INDEX_TYPE dk = -1; dk < 1; ++dk )
              {
                if( k + dk < 0 || k + dk >= m_numElemsZ ) continue;

                INDEX_TYPE const elemIndex = ( i + di ) + elemJp * ( j + dj ) + elemKp * ( k + dk );
                m_nodeToElemMap.emplaceBack( nodeIndex, elemIndex );
              }
            }
          }
        }
      }
    }

    for( INDEX_TYPE nodeID = 0; nodeID < m_numNodes; ++nodeID )
    { sortedArrayManipulation::makeSorted( m_nodeToElemMap[ nodeID ].begin(), m_nodeToElemMap[ nodeID ].end() ); }

    m_nodeToElemMap.compress();
  }

  ~SparsityGenerationNative()
  {
    CALI_CXX_MARK_SCOPE( "~SparsityGenerationNative" );

    if( !m_destructorCheck ) return;

    INDEX_TYPE const nodeJp = m_numElemsX + 1;
    INDEX_TYPE const nodeKp = nodeJp * ( m_numElemsY + 1 );

    /// Iterate over all the nodes.
    m_sparsity.move( MemorySpace::host );

    #if defined(RAJA_ENABLE_OPENMP)
    using EXEC_POLICY = parallelHostPolicy;
    #else
    using EXEC_POLICY = serialPolicy;
    #endif

    forall< EXEC_POLICY >( m_numNodes, [&] ( INDEX_TYPE const nodeID )
    {
      INDEX_TYPE const k = nodeID / nodeKp;
      INDEX_TYPE const tmp = nodeID - nodeKp * k;
      INDEX_TYPE const j = tmp / nodeJp;
      INDEX_TYPE const i = tmp - nodeJp * j;

      /// Iterate over all the neighbors of the node.
      INDEX_TYPE numNeighbors = 0;
      for( INDEX_TYPE di = -1; di < 2; ++di )
      {
        if( i + di < 0 || i + di >= m_numElemsX + 1 ) continue;
        for( INDEX_TYPE dj = -1; dj < 2; ++dj )
        {
          if( j + dj < 0 || j + dj >= m_numElemsY + 1 ) continue;
          for( INDEX_TYPE dk = -1; dk < 2; ++dk )
          {
            if( k + dk < 0 || k + dk >= m_numElemsZ + 1 ) continue;

            ++numNeighbors;
            INDEX_TYPE const nodeIndex1 = ( i + di ) + nodeJp * ( j + dj ) + nodeKp * ( k + dk );

            for( int dim0 = 0; dim0 < NDIM; ++dim0 )
            {
              for( int dim1 = 0; dim1< NDIM; ++dim1 )
              {
                LVARRAY_ERROR_IF( m_sparsity.empty( NDIM * nodeID + dim0, NDIM * nodeIndex1 + dim1 ), "Should not be empty!" );
              }
            }
          }
        }
      }

      for( int dim = 0; dim < NDIM; ++dim )
      { LVARRAY_ERROR_IF_NE( m_sparsity.numNonZeros( NDIM * nodeID + dim ), NDIM * numNeighbors ); }
    } );

    m_state.counters[ "Entries Inserted"] = ::benchmark::Counter( m_sparsity.numNonZeros(), benchmark::Counter::kIsIterationInvariantRate,
                                                                  benchmark::Counter::OneK::kIs1000 );
  }

  void resize( INDEX_TYPE const initialCapacity );

  void resizeExact();

  void generateElemLoop() const
  { generateElemLoop( m_sparsity.toView(), m_elemToNodeMap.toViewConst() ); }

  void generateNodeLoop() const
  { generateNodeLoop( m_sparsity.toView(), m_elemToNodeMap.toViewConst(), m_nodeToElemMap.toViewConst() ); }

protected:

  template< typename POLICY >
  void resizeFromNNZPerRow( std::vector< INDEX_TYPE > const & nnzPerRow );

  static void generateElemLoop( SparsityPatternViewT const & sparsity,
                                ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap );

  static void generateNodeLoop( SparsityPatternViewT const & sparsity,
                                ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                                ArrayOfArraysViewT< INDEX_TYPE const, true > const & nodeToElemMap );

  ::benchmark::State & m_state;

  INDEX_TYPE m_numElemsX;
  INDEX_TYPE m_numElemsY;
  INDEX_TYPE m_numElemsZ;
  INDEX_TYPE m_numElems;
  INDEX_TYPE m_numNodes;

  ArrayT< INDEX_TYPE, ELEM_TO_NODE_PERM > m_elemToNodeMap;
  ArrayOfArraysT< INDEX_TYPE > m_nodeToElemMap;
  SparsityPatternT m_sparsity;
  bool const m_destructorCheck;
};

template< typename POLICY >
class SparsityGenerationRAJA : public SparsityGenerationNative
{
public:

  SparsityGenerationRAJA( ::benchmark::State & state, bool const destructorCheck=true ):
    SparsityGenerationNative( state, destructorCheck )
  {
    m_elemToNodeMap.move( RAJAHelper< POLICY >::space, false );
    m_nodeToElemMap.move( RAJAHelper< POLICY >::space, false );
  }

  void generateNodeLoop() const
  { generateNodeLoop( m_sparsity.toView(), m_elemToNodeMap.toViewConst(), m_nodeToElemMap.toViewConst(), m_state ); }

  void resizeExact();

  // Note this shoule be protected but cuda won't let you put an extended lambda in a protected or private method.
  static void generateNodeLoop( SparsityPatternViewT const & sparsity,
                                ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                                ArrayOfArraysViewT< INDEX_TYPE const, true > const & nodeToElemMap,
                                ::benchmark::State & state );
};

template< typename POLICY >
class CRSMatrixAddToRow : public SparsityGenerationRAJA< POLICY >
{
public:
  CRSMatrixAddToRow( ::benchmark::State & state ):
    SparsityGenerationRAJA< POLICY >( state, false )
  {
    this->resizeExact();

    #if defined(RAJA_ENABLE_OPENMP)
    using EXEC_POLICY = parallelHostPolicy;
    #else
    using EXEC_POLICY = serialPolicy;
    #endif

    SparsityGenerationRAJA< EXEC_POLICY >::generateNodeLoop( this->m_sparsity.toView(),
                                                             this->m_elemToNodeMap.toViewConst(),
                                                             this->m_nodeToElemMap.toViewConst(),
                                                             this->m_state );

    m_matrix.assimilate< EXEC_POLICY >( std::move( this->m_sparsity ) );
    m_matrix.toViewConstSizes().move( RAJAHelper< POLICY >::space );
  }

  ~CRSMatrixAddToRow()
  {
    CALI_CXX_MARK_SCOPE( "~CRSMatrixAddToRow" );

    INDEX_TYPE const nodeJp = this->m_numElemsX + 1;
    INDEX_TYPE const nodeKp = nodeJp * ( this->m_numElemsY + 1 );

    m_matrix.move( MemorySpace::host, false );
    this->m_nodeToElemMap.move( MemorySpace::host, false );

    #if defined(RAJA_ENABLE_OPENMP)
    using EXEC_POLICY = parallelHostPolicy;
    #else
    using EXEC_POLICY = serialPolicy;
    #endif

    /// Iterate over all the nodes.
    forall< EXEC_POLICY >( this->m_numNodes, [&] ( INDEX_TYPE const nodeID )
    {
      INDEX_TYPE const k = nodeID / nodeKp;
      INDEX_TYPE const tmp = nodeID - nodeKp * k;
      INDEX_TYPE const j = tmp / nodeJp;
      INDEX_TYPE const i = tmp - nodeJp * j;

      /// Iterate over all the neighbors of the node.
      INDEX_TYPE numNeighbors = 0;
      for( INDEX_TYPE di = -1; di < 2; ++di )
      {
        if( i + di < 0 || i + di >= this->m_numElemsX + 1 ) continue;
        for( INDEX_TYPE dj = -1; dj < 2; ++dj )
        {
          if( j + dj < 0 || j + dj >= this->m_numElemsY + 1 ) continue;
          for( INDEX_TYPE dk = -1; dk < 2; ++dk )
          {
            if( k + dk < 0 || k + dk >= this->m_numElemsZ + 1 ) continue;

            ++numNeighbors;
            INDEX_TYPE const nodeIndex1 = ( i + di ) + nodeJp * ( j + dj ) + nodeKp * ( k + dk );


            std::vector< INDEX_TYPE > sharedElems;
            std::set_intersection( this->m_nodeToElemMap[ nodeID ].begin(), this->m_nodeToElemMap[ nodeID ].end(),
                                   this->m_nodeToElemMap[ nodeIndex1 ].begin(), this->m_nodeToElemMap[ nodeIndex1 ].end(),
                                   std::back_inserter( sharedElems ) );
            INDEX_TYPE const numSharedElems = sharedElems.size();

            for( int dim0 = 0; dim0 < NDIM; ++dim0 )
            {
              INDEX_TYPE const dof0 = NDIM * nodeID + dim0;
              for( int dim1 = 0; dim1< NDIM; ++dim1 )
              {
                INDEX_TYPE const dof1 = NDIM * nodeIndex1 + dim1;
                LVARRAY_ERROR_IF( m_matrix.empty( dof0, dof1 ), "Should not be empty!" );

                COLUMN_TYPE const * const columns = m_matrix.getColumns( dof0 );
                INDEX_TYPE const pos = sortedArrayManipulation::find( columns, m_matrix.numNonZeros( dof0 ), dof1 );

                if( !equal( m_matrix.getEntries( dof0 )[ pos ], ENTRY_TYPE( ( dof0 - dof1 ) * numSharedElems * this->m_state.iterations() ) ) )
                {
                  LVARRAY_LOG( "m_matrix.getEntries( dof0 )[ pos ] ) = " << m_matrix.getEntries( dof0 )[ pos ] << "\n" <<
                               "ENTRY_TYPE( ( dof0 - dof1 ) ) * numSharedElems * this->m_state.iterations() = " << ENTRY_TYPE( ( dof0 - dof1 ) ) *
                               numSharedElems * this->m_state.iterations() << "\n" <<
                               "dof0 = " << dof0 << "\n" <<
                               "dof1 = " << dof1 << "\n" <<
                               "numSharedElems = " << numSharedElems << "\n" <<
                               "state.iterations() = " << this->m_state.iterations() << "\n" );
                }
              }
            }
          }
        }
      }

      for( int dim = 0; dim < NDIM; ++dim )
      { LVARRAY_ERROR_IF_NE( m_matrix.numNonZeros( NDIM * nodeID + dim ), NDIM * numNeighbors ); }
    } );

    this->m_state.counters[ "Additions"] = ::benchmark::Counter( this->m_numElems * NODES_PER_ELEM * NODES_PER_ELEM * NDIM * NDIM,
                                                                 benchmark::Counter::kIsIterationInvariantRate,
                                                                 benchmark::Counter::OneK::kIs1000 );
  }

  void add() const
  { addKernel( m_matrix.toViewConstSizes(), this->m_elemToNodeMap.toViewConst() ); }

  // Note this shoule be protected but cuda won't let you put an extended lambda in a protected or private method.
  static void addKernel( CRSMatrixViewConstSizesT const & matrix,
                         ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap );

private:
  CRSMatrixT m_matrix;
};

} // namespace benchmarking
} // namespace LvArray
