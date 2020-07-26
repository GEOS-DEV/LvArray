// Source includes
#include "testHelpers.hpp"
#include "../../src/jitti/jitti.hpp"

// TPL includes
#include <gtest/gtest.h>

#define USE_TEMPLATE_COMPILER_LIBS 0

TEST( TypedDynamicLibrary, serial )
{
#if !USE_TEMPLATE_COMPILER_LIBS
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestTypedDynamicLibrarySerial.so" );
  SquareAllType const squareAllSerial = dl.getSymbol< SquareAllType >( "squareAllSerial" );
#else
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libsquareAllJITSerial.so" );
  SquareAllType const squareAllSerial = dl.getSymbol< SquareAllType >( "squareAll< RAJA::loop_exec >" );
#endif

  test( squareAllSerial, "RAJA::policy::loop::loop_exec" );
}

#if defined(USE_OPENMP)
TEST( TypedDynamicLibrary, OpenMP )
{
#if !USE_TEMPLATE_COMPILER_LIBS
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestTypedDynamicLibraryOpenMP.so" );
  SquareAllType const squareAllOpenMP = dl.getSymbol< SquareAllType >( "squareAllOpenMP" );
#else
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libsquareAllJITOpenMP.so" );
  SquareAllType const squareAllOpenMP = dl.getSymbol< SquareAllType >( "squareAll< RAJA::omp_parallel_for_exec >" );
#endif

  test( squareAllOpenMP, "RAJA::policy::omp::omp_parallel_for_exec" );
}
#endif

#if defined(USE_CUDA)
TEST( TypedDynamicLibrary, CUDA )
{
#if !USE_TEMPLATE_COMPILER_LIBS
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libtestTypedDynamicLibraryCUDA.so" );
  SquareAllType const squareAllCUDA = dl.getSymbol< SquareAllType >( "squareAllCUDA" );
#else
  jitti::TypedDynamicLibrary dl( JITTI_OUTPUT_DIR "/libsquareAllJITCUDA.so" );
  SquareAllType const squareAllCUDA = dl.getSymbol< SquareAllType >( "squareAll< RAJA::cuda_exec< 32 > >" );
#endif

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

