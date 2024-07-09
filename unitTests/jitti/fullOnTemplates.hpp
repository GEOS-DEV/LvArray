#pragma once

#include "ArrayView.hpp"
#include "ChaiBuffer.hpp"

#include <umpire/ResourceManager.hpp>

constexpr auto fullOnTemplatesPath = __FILE__;

template< typename T >
std::string getUmpireAllocatorName( T * const ptr )
{
  return umpire::ResourceManager::getInstance().getAllocator( ptr ).getName();
}

template< typename POLICY >
void cubeAll( LvArray::ArrayView< int, 1, 0, int, LvArray::ChaiBuffer > const & dst,
              LvArray::ArrayView< int const, 1, 0, int, LvArray::ChaiBuffer > const & src )
{
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< int >( 0, dst.size() ),
    [dst, src] LVARRAY_HOST_DEVICE ( int const i )
    {
      dst[ i ] = src[ i ] * src[ i ] * src[ i ];
    }
  );
}
