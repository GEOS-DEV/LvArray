// Source includes
#include "testHelpers.hpp"
#include "../../src/jitti/jitti.hpp"

// TPL includes
#include <gtest/gtest.h>

TEST( DynamicLibrary, serial )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibrarySerial.so" );
  SquareAllType const squareAllSerial = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllSerial" ) );
  test( squareAllSerial, "RAJA::policy::loop::loop_exec" );
}

#if defined(USE_OPENMP)
TEST( DynamicLibrary, OpenMP )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibraryOpenMP.so" );
  SquareAllType const squareAllOpenMP = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllOpenMP" ) );
  test( squareAllOpenMP, "RAJA::policy::omp::omp_parallel_for_exec" );
}
#endif

#if defined(USE_CUDA)
TEST( DynamicLibrary, CUDA )
{
  jitti::DynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestDynamicLibraryCUDA.so" );
  SquareAllType const squareAllCUDA = reinterpret_cast< SquareAllType >( dl.getSymbol( "squareAllCUDA" ) );
  test( squareAllCUDA, "RAJA::policy::cuda::cuda_exec<32ul, false>" );
}
#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

