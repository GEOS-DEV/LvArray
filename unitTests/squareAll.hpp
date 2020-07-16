#pragma once

// Source includes
#include "ArraySlice.hpp"
#include "StringUtilities.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <string>
#include <unordered_map>
#include <typeindex>

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

template< typename T >
std::pair< std::string, std::pair< void *, std::type_index > >
createEntry( std::string const & name, T * symbol )
{ return { name, { reinterpret_cast< void * >( symbol ), std::type_index( typeid( symbol ) ) } }; }

namespace internal
{

template< typename POLICY >
void squareAll( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const dst,
                LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const src )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  LVARRAY_LOG( "Using POLICY = " << LvArray::demangleType< POLICY >() );
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, dst.size() ),
    [dst, src] ( std::ptrdiff_t const i)
    {
      dst[ i ] = src[ i ] * src[ i ];
    }
  );
}

} // namespace internal


