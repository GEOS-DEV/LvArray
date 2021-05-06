/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkSparsityGenerationKernels.hpp"

// TPL includes
#include <benchmark/benchmark.h>

// System includes
#include <utility>

namespace LvArray
{
namespace benchmarking
{

LVARRAY_HOST_DEVICE
INDEX_TYPE getNeighborNodes( INDEX_TYPE (& neighborNodes)[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ],
                             ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                             ArraySliceT< INDEX_TYPE const, RAJA::PERM_I > const nodeElems )
{
  INDEX_TYPE numNeighbors = 0;
  for( INDEX_TYPE localElem = 0; localElem < nodeElems.size(); ++localElem )
  {
    INDEX_TYPE const elemID = nodeElems[ localElem ];
    for( INDEX_TYPE localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      neighborNodes[ numNeighbors++ ] = elemToNodeMap( elemID, localNode );
    }
  }

  return sortedArrayManipulation::makeSortedUnique( neighborNodes, neighborNodes + numNeighbors );
}


DISABLE_HD_WARNING
template< typename SPARSITY_TYPE >
LVARRAY_HOST_DEVICE
void insertEntriesForNode( SPARSITY_TYPE & sparsity,
                           INDEX_TYPE const nodeID,
                           ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                           ArraySliceT< INDEX_TYPE const, RAJA::PERM_I > const nodeElems )
{
  INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
  INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, elemToNodeMap, nodeElems );

  INDEX_TYPE dofNumbers[ MAX_COLUMNS_PER_ROW ];
  for( INDEX_TYPE i = 0; i < numNeighbors; ++i )
  {
    for( INDEX_TYPE dim = 0; dim < NDIM; ++dim )
    {
      dofNumbers[ NDIM * i + dim ] = NDIM * neighborNodes[ i ] + dim;
    }
  }

  for( int dim = 0; dim < NDIM; ++dim )
  {
    sparsity.insertNonZeros( NDIM * nodeID + dim, dofNumbers, dofNumbers + NDIM * numNeighbors );
  }
}


void SparsityGenerationNative::resize( INDEX_TYPE const initialCapacity )
{
  CALI_CXX_MARK_SCOPE( "resize" );
  m_sparsity = SparsityPatternT( NDIM * m_numNodes, NDIM * m_numNodes, initialCapacity );
}

void SparsityGenerationNative::resizeExact()
{
  std::vector< INDEX_TYPE > nnzPerRow( 3 * m_numNodes );

  INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
  for( INDEX_TYPE nodeID = 0; nodeID < m_numNodes; ++nodeID )
  {
    INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, m_elemToNodeMap.toViewConst(), m_nodeToElemMap[ nodeID ] );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      nnzPerRow[ NDIM * nodeID + dim ] = NDIM * numNeighbors;
    }
  }

  resizeFromNNZPerRow< serialPolicy >( nnzPerRow );
}

template< typename POLICY >
void SparsityGenerationNative::resizeFromNNZPerRow( std::vector< INDEX_TYPE > const & nnzPerRow )
{
  SparsityPatternT newSparsity;
  newSparsity.resizeFromRowCapacities< POLICY >( NDIM * m_numNodes, NDIM * m_numNodes, nnzPerRow.data() );
  m_sparsity = std::move( newSparsity );
}

void SparsityGenerationNative::generateElemLoop( SparsityPatternViewT const & sparsity,
                                                 ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap )
{
  COLUMN_TYPE dofNumbers[ NODES_PER_ELEM * NDIM ];
  for( INDEX_TYPE elemID = 0; elemID < elemToNodeMap.size( 0 ); ++elemID )
  {
    for( INDEX_TYPE localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      for( int dim = 0; dim < NDIM; ++dim )
      {
        dofNumbers[ NDIM * localNode + dim ] = NDIM * elemToNodeMap( elemID, localNode ) + dim;
      }
    }

    sortedArrayManipulation::makeSorted( &dofNumbers[ 0 ], &dofNumbers[ NODES_PER_ELEM * NDIM ] );

    for( int localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
    {
      for( int dim = 0; dim < NDIM; ++dim )
      {
        sparsity.insertNonZeros( dofNumbers[ NDIM * localNode + dim ], &dofNumbers[ 0 ], &dofNumbers[ NODES_PER_ELEM * NDIM ] );
      }
    }
  }
}

void SparsityGenerationNative::generateNodeLoop( SparsityPatternViewT const & sparsity,
                                                 ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                                                 ArrayOfArraysViewT< INDEX_TYPE const, true > const & nodeToElemMap )
{
  /// Iterate over all the nodes.
  for( INDEX_TYPE nodeID = 0; nodeID < nodeToElemMap.size(); ++nodeID )
  {
    insertEntriesForNode( sparsity, nodeID, elemToNodeMap, nodeToElemMap[ nodeID ] );
  }
}

template< typename POLICY >
void SparsityGenerationRAJA< POLICY >::
resizeExact()
{
  CALI_CXX_MARK_SCOPE( "resizeExact" );
  std::vector< INDEX_TYPE > nnzPerRow( 3 * m_numNodes );

  #if defined(RAJA_ENABLE_OPENMP)
  using RESIZE_POLICY = std::conditional_t< std::is_same< serialPolicy, POLICY >::value, serialPolicy, parallelHostPolicy >;
  #else
  using RESIZE_POLICY = serialPolicy;
  #endif

  ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const elemToNodeMap = m_elemToNodeMap.toViewConst();
  ArrayOfArraysViewT< INDEX_TYPE const, true > const nodeToElemMap = m_nodeToElemMap.toViewConst();
  forall< RESIZE_POLICY >( m_numNodes, [&nnzPerRow, elemToNodeMap, nodeToElemMap] ( INDEX_TYPE const nodeID )
  {
    INDEX_TYPE neighborNodes[ MAX_ELEMS_PER_NODE * NODES_PER_ELEM ];
    INDEX_TYPE const numNeighbors = getNeighborNodes( neighborNodes, elemToNodeMap, nodeToElemMap[ nodeID ] );
    for( int dim = 0; dim < NDIM; ++dim )
    {
      nnzPerRow[ NDIM * nodeID + dim ] = NDIM * numNeighbors;
    }
  } );

  resizeFromNNZPerRow< RESIZE_POLICY >( nnzPerRow );
}

// Note this shoule be protected but cuda won't let you put an extended lambda in a protected or private method.
template< typename POLICY >
void SparsityGenerationRAJA< POLICY >::
generateNodeLoop( SparsityPatternViewT const & sparsity,
                  ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap,
                  ArrayOfArraysViewT< INDEX_TYPE const, true > const & nodeToElemMap,
                  ::benchmark::State & state )
{
  CALI_CXX_MARK_SCOPE( "generateNodeLoop" );

  // This isn't measured for the other benchmarks so it's not here either.
  if( state.iterations() )
  {
    state.PauseTiming();
  }

  sparsity.move( RAJAHelper< POLICY >::space );

  if( state.iterations() )
  {
    state.ResumeTiming();
  }

  forall< POLICY >( nodeToElemMap.size(), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const nodeID )
      {
        insertEntriesForNode( sparsity, nodeID, elemToNodeMap, nodeToElemMap[ nodeID ] );
      } );
}

template< typename POLICY >
void CRSMatrixAddToRow< POLICY >::
addKernel( CRSMatrixViewConstSizesT const & matrix,
           ArrayViewT< INDEX_TYPE const, ELEM_TO_NODE_PERM > const & elemToNodeMap )
{
  CALI_CXX_MARK_SCOPE( "addKernel" );

  forall< POLICY >( elemToNodeMap.size( 0 ), [matrix, elemToNodeMap] LVARRAY_HOST_DEVICE ( INDEX_TYPE const elemID )
      {
        COLUMN_TYPE dofNumbers[ NODES_PER_ELEM * NDIM ];
        ENTRY_TYPE additions[ NODES_PER_ELEM * NDIM ][ NODES_PER_ELEM * NDIM ];
        for( INDEX_TYPE localNode0 = 0; localNode0 < NODES_PER_ELEM; ++localNode0 )
        {
          for( int dim0 = 0; dim0 < NDIM; ++dim0 )
          {
            INDEX_TYPE const dof0 = NDIM * elemToNodeMap( elemID, localNode0 ) + dim0;
            dofNumbers[ NDIM * localNode0 + dim0 ] = dof0;

            for( INDEX_TYPE localNode1 = 0; localNode1 < NODES_PER_ELEM; ++localNode1 )
            {
              for( int dim1 = 0; dim1 < NDIM; ++dim1 )
              {
                INDEX_TYPE const dof1 = NDIM * elemToNodeMap( elemID, localNode1 ) + dim1;
                additions[ NDIM * localNode0 + dim0][ NDIM * localNode1 + dim1 ] = dof0 - dof1;
              }
            }

          }
        }

        for( int localNode = 0; localNode < NODES_PER_ELEM; ++localNode )
        {
          for( int dim = 0; dim < NDIM; ++dim )
          {
            matrix.addToRowBinarySearchUnsorted< typename RAJAHelper< POLICY >::AtomicPolicy >( dofNumbers[ NDIM * localNode + dim ], dofNumbers,
                                                                                                additions[ NDIM * localNode + dim ], NODES_PER_ELEM * NDIM );
          }
        }
      } );
}

// Explicit instantiation of SparsityGenerationRAJA.
template class SparsityGenerationRAJA< serialPolicy >;
template class CRSMatrixAddToRow< serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class SparsityGenerationRAJA< parallelHostPolicy >;
template class CRSMatrixAddToRow< parallelHostPolicy >;
#endif

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
template class SparsityGenerationRAJA< parallelDevicePolicy< THREADS_PER_BLOCK > >;
template class CRSMatrixAddToRow< parallelDevicePolicy< THREADS_PER_BLOCK > >;
#endif

} // namespace benchmarking
} // namespace LvArray
