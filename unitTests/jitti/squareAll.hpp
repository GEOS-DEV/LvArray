// Source includes
#include "../../src/ArrayView.hpp"
#include "../../src/NewChaiBuffer.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

template< typename POLICY >
void squareAll( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const & dst,
                LvArray::ArrayView< int const, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const & src,
                std::string & policy )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  policy = LvArray::demangleType< POLICY >();
  LVARRAY_LOG( "squareAll called with " << policy );

  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, dst.size() ),
    [dst, src] LVARRAY_HOST_DEVICE ( std::ptrdiff_t const i)
    {
      dst[ i ] = src[ i ] * src[ i ];
    }
  );
}
