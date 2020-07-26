// Source includes
#include "../../src/ArraySlice.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

extern "C"
void squareAllOpenMP( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const dst,
                      LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const src,
                      std::string & policy )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  #define POLICY RAJA::omp_parallel_for_exec
  LVARRAY_LOG( "squareAllOpenMP called, uses " STRINGIZE( POLICY ) );
  policy = STRINGIZE( POLICY );

  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, dst.size() ),
    [dst, src] ( std::ptrdiff_t const i)
    {
      dst[ i ] = src[ i ] * src[ i ];
    }
  );
}
