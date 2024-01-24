#include "Array.hpp"
#include "ArrayOfArrays.hpp"
#include <RAJA/util/Permutations.hpp>

#include "MallocBuffer.hpp"


void breakpoint_helper()
{
  // Left blank. This function is there to be used as a breakpoint for gdb.
}


template< typename T >
using array1d = LvArray::Array< int, 1, RAJA::PERM_I, int, LvArray::MallocBuffer >;
template< typename T >
using array2d = LvArray::Array< int, 2, RAJA::PERM_IJ, int, LvArray::MallocBuffer >;
template< typename T >
using array3d = LvArray::Array< int, 3, RAJA::PERM_IJK, int, LvArray::MallocBuffer >;
template< typename T >
using aoa = LvArray::ArrayOfArrays< T, int, LvArray::MallocBuffer >;


int main()
{
  array1d< int > v0;

  array1d< int > v1;
  v1.emplace_back( 1 );
  v1.emplace_back( 2 );

  auto v1v = v1.toView();
  LVARRAY_LOG( v1v[0] );  // Calling `LVARRAY_LOG` to prevent the variable from being unused.
  auto v1vc = v1.toViewConst();
  LVARRAY_LOG( v1vc[0] );

  array2d< int > v2( 2, 3 );
  v2[0][0] = 1;
  v2[0][1] = 2;
  v2[0][2] = 3;
  v2[1][0] = 4;
  v2[1][1] = 5;
  v2[1][2] = 6;

  auto v2v = v2.toView();
  LVARRAY_LOG( v2v[0][0] );
  auto v2vc = v2.toViewConst();
  LVARRAY_LOG( v2vc[0][0] );
  auto v2s = v2[0];
  LVARRAY_LOG( v2s[0] );
  auto v2sc = v2[0].toSliceConst();
  LVARRAY_LOG( v2sc[0] );

  array3d< int > v3( 2, 3, 4 );
  for( int i = 0, count = 0; i < 2; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      for( int k = 0; k < 4; ++k, ++count )
      {
        v3[i][j][k] = count;
      }
    }
  }

  auto v3v = v3.toView();
  LVARRAY_LOG( v3v[0][0][0] );
  auto v3vc = v3.toViewConst();
  LVARRAY_LOG( v3vc[0][0][0] );

  auto v3s = v3[0];
  LVARRAY_LOG( v3s[0][0] );
  auto v3s2 = v3[0][0];
  LVARRAY_LOG( v3s2[0] );

  aoa< int > aoa0( 2, 10 );
  aoa0.emplaceBack( 0, 1 );
  aoa0.emplaceBack( 0, 2 );
  aoa0.emplaceBack( 0, 3 );
  aoa0.emplaceBack( 1, 7 );
  aoa0.emplaceBack( 1, 8 );

  auto aoa0v = aoa0.toView();
  LVARRAY_LOG( aoa0v[0][0] );
  auto aoa0vc = aoa0.toViewConst();
  LVARRAY_LOG( aoa0vc[0][0] );

  auto aoa0s = aoa0[1];
  LVARRAY_LOG( aoa0s[0] );

  breakpoint_helper();

  return 0;
}