// Source includes
#include "../../src/Array.hpp"
#include "../../src/NewChaiBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

using SquareAllType = void (*)( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const,
                                LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const,
                                std::string & );

void test( SquareAllType squareAll, std::string const & expectedPolicy )
{
  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::NewChaiBuffer > output( 100 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::NewChaiBuffer > input( 100 );

  for ( std::ptrdiff_t i = 0; i < input.size(); ++i )
  { input[ i ] = i; }

  // Call the function.
  std::string policy;
  squareAll( output, input, policy );
  EXPECT_EQ( policy, expectedPolicy );

  // Check the output.
  for ( std::ptrdiff_t i = 0; i < output.size(); ++i )
  { EXPECT_EQ( output[ i ], i * i ); }
}
