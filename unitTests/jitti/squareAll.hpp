// Source includes
#include "../../src/ArrayView.hpp"
#include "../../src/ChaiBuffer.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>
#include <umpire/ResourceManager.hpp>

template< typename POLICY >
void squareAll( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer > const & dst,
                LvArray::ArrayView< int const, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer > const & src )
{
  LVARRAY_LOG( "In " << __PRETTY_FUNCTION__ );
  
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  auto allocator = umpire::ResourceManager::getInstance().getAllocator( dst.data() );
  LVARRAY_LOG( allocator.getName() );

  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, dst.size() ),
    [dst, src] LVARRAY_HOST_DEVICE ( std::ptrdiff_t const i )
    {
      dst[ i ] = src[ i ] * src[ i ];
    }
  );
}
