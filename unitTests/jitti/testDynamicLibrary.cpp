// Source includes
#include "../../src/jitti/jitti.hpp"
#include "../../src/Array.hpp"
#include "../../src/MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

using SquareAllType = void (*)( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const,
                                LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const,
                                std::string & policy );

void test( SquareAllType squareAll, std::string const & expectedPolicy )
{
  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > output( 100 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > input( 100 );

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

TEST( DynamicLibrary, serial )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibrarySerial.so" );
  SquareAllType const squareAllSerial = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllSerial" ) );
  test( squareAllSerial, "RAJA::loop_exec" );
}

#if defined(USE_OPENMP)
TEST( DynamicLibrary, OpenMP )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibraryOpenMP.so" );
  SquareAllType const squareAllOpenMP = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllOpenMP" ) );
  test( squareAllOpenMP, "RAJA::omp_parallel_for_exec" );
}
#endif

#if defined(USE_CUDA)
TEST( DynamicLibrary, CUDA )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibraryCUDA.so" );
  SquareAllType const squareAllCUDA = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllCUDA" ) );
  test( squareAllCUDA, "RAJA::cuda_exec< 32 >" );
}
#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  jitti::getCompileTime();
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

