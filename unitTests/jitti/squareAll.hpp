#pragma once

// Source includes
#include "../../src/ArraySlice.hpp"
#include "../../src/jitti/jitti.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

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

jitti::CompilationInfo getCompilationInfo();
