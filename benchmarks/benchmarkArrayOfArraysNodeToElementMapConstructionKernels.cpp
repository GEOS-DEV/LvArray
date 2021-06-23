/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkArrayOfArraysNodeToElementMapConstructionKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

// Sphinx start after vector
void NaiveNodeToElemMapConstruction::
  vector( ArrayView< INDEX_TYPE const, 2, 1, INDEX_TYPE, DEFAULT_BUFFER > const & elementToNodeMap,
          std::vector< std::vector< INDEX_TYPE > > & nodeToElementMap,
          INDEX_TYPE const numNodes )
{
  nodeToElementMap.resize( numNodes );

  for( INDEX_TYPE elementIndex = 0; elementIndex < elementToNodeMap.size( 0 ); ++elementIndex )
  {
    for( INDEX_TYPE const nodeIndex : elementToNodeMap[ elementIndex ] )
    {
      nodeToElementMap[ nodeIndex ].emplace_back( elementIndex );
    }
  }
}
// Sphinx end before vector

// Sphinx start after naive
void NaiveNodeToElemMapConstruction::
  naive( ArrayView< INDEX_TYPE const, 2, 1, INDEX_TYPE, DEFAULT_BUFFER > const & elementToNodeMap,
         ArrayOfArrays< INDEX_TYPE, INDEX_TYPE, DEFAULT_BUFFER > & nodeToElementMap,
         INDEX_TYPE const numNodes )
{
  nodeToElementMap.resize( numNodes );

  for( INDEX_TYPE elementIndex = 0; elementIndex < elementToNodeMap.size( 0 ); ++elementIndex )
  {
    for( INDEX_TYPE const nodeIndex : elementToNodeMap[ elementIndex ] )
    {
      nodeToElementMap.emplaceBack( nodeIndex, elementIndex );
    }
  }
}
// Sphinx end before naive

// Sphinx start after overAllocation
template< typename POLICY >
void NodeToElemMapConstruction< POLICY >::
overAllocation( ArrayView< INDEX_TYPE const, 2, 1, INDEX_TYPE, DEFAULT_BUFFER > const & elementToNodeMap,
                ArrayOfArrays< INDEX_TYPE, INDEX_TYPE, DEFAULT_BUFFER > & nodeToElementMap,
                INDEX_TYPE const numNodes,
                INDEX_TYPE const maxNodeElements )
{
  using ATOMIC_POLICY = typename RAJAHelper< POLICY >::AtomicPolicy;

  // Resize the node to element map allocating space for each inner array.
  nodeToElementMap.resize( numNodes, maxNodeElements );

  // Create an ArrayOfArraysView
  ArrayOfArraysView< INDEX_TYPE, INDEX_TYPE const, false, DEFAULT_BUFFER > const nodeToElementMapView =
    nodeToElementMap.toView();

  // Launch a RAJA kernel that populates the ArrayOfArraysView.
  RAJA::forall< POLICY >(
    RAJA::TypedRangeSegment< INDEX_TYPE >( 0, elementToNodeMap.size( 0 ) ),
    [elementToNodeMap, nodeToElementMapView] ( INDEX_TYPE const elementIndex )
  {
    for( INDEX_TYPE const nodeIndex : elementToNodeMap[ elementIndex ] )
    {
      nodeToElementMapView.emplaceBackAtomic< ATOMIC_POLICY >( nodeIndex, elementIndex );
    }
  }
    );
}
// Sphinx end before overAllocation

// Sphinx start after resizeFromCapacities
template< typename POLICY >
void NodeToElemMapConstruction< POLICY >::
resizeFromCapacities( ArrayView< INDEX_TYPE const, 2, 1, INDEX_TYPE, DEFAULT_BUFFER > const & elementToNodeMap,
                      ArrayOfArrays< INDEX_TYPE, INDEX_TYPE, DEFAULT_BUFFER > & nodeToElementMap,
                      INDEX_TYPE const numNodes )
{
  using ATOMIC_POLICY = typename RAJAHelper< POLICY >::AtomicPolicy;

  // Create an Array containing the size of each inner array.
  Array< INDEX_TYPE, 1, RAJA::PERM_I, INDEX_TYPE, DEFAULT_BUFFER > elementsPerNode( numNodes );

  // Calculate the size of each inner array.
  RAJA::forall< POLICY >(
    RAJA::TypedRangeSegment< INDEX_TYPE >( 0, elementToNodeMap.size( 0 ) ),
    [elementToNodeMap, &elementsPerNode] ( INDEX_TYPE const elementIndex )
  {
    for( INDEX_TYPE const nodeIndex : elementToNodeMap[ elementIndex ] )
    {
      RAJA::atomicInc< ATOMIC_POLICY >( &elementsPerNode[ nodeIndex ] );
    }
  }
    );

  // Resize the node to element map with the inner array sizes.
  nodeToElementMap.resizeFromCapacities< POLICY >( elementsPerNode.size(), elementsPerNode.data() );

  // Create an ArrayOfArraysView
  ArrayOfArraysView< INDEX_TYPE, INDEX_TYPE const, false, DEFAULT_BUFFER > const nodeToElementMapView =
    nodeToElementMap.toView();

  // Launch a RAJA kernel that populates the ArrayOfArraysView.
  RAJA::forall< POLICY >(
    RAJA::TypedRangeSegment< INDEX_TYPE >( 0, elementToNodeMap.size( 0 ) ),
    [elementToNodeMap, nodeToElementMapView] ( INDEX_TYPE const elementIndex )
  {
    for( INDEX_TYPE const nodeIndex : elementToNodeMap[ elementIndex ] )
    {
      nodeToElementMapView.emplaceBackAtomic< ATOMIC_POLICY >( nodeIndex, elementIndex );
    }
  } );
}
// Sphinx end before resizeFromCapacities

// Explicit instantiation of NodeToElemMapConstruction.
template class NodeToElemMapConstruction< serialPolicy >;

#if defined(RAJA_ENABLE_OPENMP)
template class NodeToElemMapConstruction< parallelHostPolicy >;
#endif

} // namespace benchmarking
} // namespace LvArray
