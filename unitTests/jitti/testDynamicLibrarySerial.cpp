// Source includes
#include "../../src/ArraySlice.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

extern "C"
void squareAllSerial( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const dst,
                      LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const src,
                      std::string & policy )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  #define POLICY RAJA::loop_exec
  LVARRAY_LOG( "squareAllSerial called, uses " STRINGIZE( POLICY ) );
  policy = STRINGIZE( POLICY );

  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, dst.size() ),
    [dst, src] ( std::ptrdiff_t const i)
    {
      dst[ i ] = src[ i ] * src[ i ];
    }
  );
}
