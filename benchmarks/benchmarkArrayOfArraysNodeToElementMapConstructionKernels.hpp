/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#pragma once

// Source includes
#include "benchmarkHelpers.hpp"

// TPL includes
#include <benchmark/benchmark.h>

namespace LvArray
{
namespace benchmarking
{

#define TIMING_LOOP( INIT, KERNEL ) \
  for( auto _ : m_state ) \
  { \
    LVARRAY_UNUSED_VARIABLE( _ ); \
    m_state.PauseTiming(); \
    INIT; \
    m_state.ResumeTiming(); \
    KERNEL; \
    ::benchmark::ClobberMemory(); \
  } \

class NaiveNodeToElemMapConstruction
{
public:

  NaiveNodeToElemMapConstruction( ::benchmark::State & state,
                                  char const * const callingFunction ):
    m_state( state ),
    m_callingFunction( callingFunction ),
    m_numElemsX( state.range( 0 ) ),
    m_numElemsY( state.range( 1 ) ),
    m_numElemsZ( state.range( 2 ) ),
    m_numNodes( ( m_numElemsX + 1 ) * ( m_numElemsY + 1 ) * ( m_numElemsZ + 1 ) ),
    m_nodeToElementMap(),
    m_elementToNodeMap()
  { constructStructuredElementToNodeMap( m_elementToNodeMap, m_numElemsX, m_numElemsY, m_numElemsZ ); }

  ~NaiveNodeToElemMapConstruction()
  {
    CALI_CXX_MARK_SCOPE( "~NaiveNodeToElemMapConstruction" );

    // #if defined(RAJA_ENABLE_OPENMP)
    //   using EXEC_POLICY = parallelHostPolicy;
    // #else
    using EXEC_POLICY = serialPolicy;
    // #endif

    RAJA::ReduceSum< typename RAJAHelper< EXEC_POLICY >::ReducePolicy, INDEX_TYPE > totalMapSize( 0 ); \

    ArrayOfArraysViewT< INDEX_TYPE, true > const nodeToElementMap = m_nodeToElementMap.toViewConstSizes();

    INDEX_TYPE const elemsX = m_numElemsX;
    INDEX_TYPE const elemsY = m_numElemsY;
    INDEX_TYPE const elemsZ = m_numElemsZ;

    INDEX_TYPE const numNodes = m_numNodes;
    forall< EXEC_POLICY >( numNodes, [totalMapSize, nodeToElementMap, elemsX, elemsY, elemsZ ] ( INDEX_TYPE const nodeIndex )
    {
      INDEX_TYPE const elemJp = elemsX;
      INDEX_TYPE const elemKp = elemJp * elemsY;

      INDEX_TYPE const nodeJp = elemsX + 1;
      INDEX_TYPE const nodeKp = nodeJp * ( elemsY + 1 );

      INDEX_TYPE const k = nodeIndex / nodeKp;
      INDEX_TYPE const tmp = nodeIndex - nodeKp * k;
      INDEX_TYPE const j = tmp / nodeJp;
      INDEX_TYPE const i = tmp - nodeJp * j;

      INDEX_TYPE elements[ 16 ];
      INDEX_TYPE numElements = 0;

      for( INDEX_TYPE di = -1; di < 1; ++di )
      {
        if( i + di < 0 || i + di >= elemsX ) continue;
        for( INDEX_TYPE dj = -1; dj < 1; ++dj )
        {
          if( j + dj < 0 || j + dj >= elemsY ) continue;
          for( INDEX_TYPE dk = -1; dk < 1; ++dk )
          {
            if( k + dk < 0 || k + dk >= elemsZ ) continue;

            INDEX_TYPE const elemIndex = ( i + di ) + elemJp * ( j + dj ) + elemKp * ( k + dk );
            elements[ numElements ] = elemIndex;
            ++numElements;
          }
        }
      }

      LVARRAY_ERROR_IF_NE_MSG( numElements, nodeToElementMap.sizeOfArray( nodeIndex ), nodeIndex );
      LVARRAY_ERROR_IF_GT( numElements, 8 );

      std::sort( elements, elements + numElements );
      std::sort( nodeToElementMap[ nodeIndex ].begin(), nodeToElementMap[ nodeIndex ].end() );

      for( INDEX_TYPE q = 0; q < numElements; ++q )
      { LVARRAY_ERROR_IF_NE( nodeToElementMap( nodeIndex, q ), elements[ q ] ); }

      totalMapSize += numElements;
    } );

    m_state.counters[ "Node to element map size"] = ::benchmark::Counter( totalMapSize.get(),
                                                                          ::benchmark::Counter::kIsIterationInvariantRate,
                                                                          ::benchmark::Counter::OneK::kIs1000 );
  }

  void vector()
  {
    std::vector< std::vector< INDEX_TYPE > > nodeToElemVector;

    TIMING_LOOP( nodeToElemVector = std::vector< std::vector< INDEX_TYPE > >(),
                 vector( m_elementToNodeMap.toViewConst(), nodeToElemVector, m_numNodes ) );

    for( std::size_t nodeID = 0; nodeID < nodeToElemVector.size(); ++nodeID )
    { m_nodeToElementMap.appendArray( nodeToElemVector[ nodeID ].begin(), nodeToElemVector[ nodeID ].end() ); }
  }

  void naive()
  {
    TIMING_LOOP( m_nodeToElementMap = ArrayOfArraysT< INDEX_TYPE >(),
                 naive( m_elementToNodeMap.toViewConst(), m_nodeToElementMap, m_numNodes ) );
  }

protected:

  ::benchmark::State & m_state;
  std::string const m_callingFunction;
  INDEX_TYPE const m_numElemsX;
  INDEX_TYPE const m_numElemsY;
  INDEX_TYPE const m_numElemsZ;
  INDEX_TYPE const m_numNodes;
  ArrayOfArraysT< INDEX_TYPE > m_nodeToElementMap;
  ArrayT< INDEX_TYPE, RAJA::PERM_IJ > m_elementToNodeMap;

private:

  static void vector( ArrayViewT< INDEX_TYPE const, RAJA::PERM_IJ > const & elementToNodeMap,
                      std::vector< std::vector< INDEX_TYPE > > & nodeToElementMap,
                      INDEX_TYPE const numNodes );

  static void naive( ArrayViewT< INDEX_TYPE const, RAJA::PERM_IJ > const & elementToNodeMap,
                     ArrayOfArraysT< INDEX_TYPE > & nodeToElementMap,
                     INDEX_TYPE const numNodes );
};

template< typename POLICY >
class NodeToElemMapConstruction : NaiveNodeToElemMapConstruction
{
public:

  NodeToElemMapConstruction( ::benchmark::State & state,
                             char const * const callingFunction ):
    NaiveNodeToElemMapConstruction( state, callingFunction )
  {}

  void overAllocation()
  {
    TIMING_LOOP( m_nodeToElementMap = ArrayOfArraysT< INDEX_TYPE >(),
                 overAllocation( m_elementToNodeMap.toViewConst(), m_nodeToElementMap, m_numNodes, 8 ) );
  }

  void resizeFromCapacities()
  {
    TIMING_LOOP( m_nodeToElementMap = ArrayOfArraysT< INDEX_TYPE >(),
                 resizeFromCapacities( m_elementToNodeMap.toViewConst(), m_nodeToElementMap, m_numNodes ) );
  }

private:

  static void overAllocation( ArrayViewT< INDEX_TYPE const, RAJA::PERM_IJ > const & elementToNodeMap,
                              ArrayOfArraysT< INDEX_TYPE > & nodeToElementMap,
                              INDEX_TYPE const numNodes,
                              INDEX_TYPE const maxNodeElements );

  static void resizeFromCapacities( ArrayViewT< INDEX_TYPE const, RAJA::PERM_IJ > const & elementToNodeMap,
                                    ArrayOfArraysT< INDEX_TYPE > & nodeToElementMap,
                                    INDEX_TYPE const numNodes );

};

#undef TIMING_LOOP

} // namespace benchmarking
} // namespace LvArray
