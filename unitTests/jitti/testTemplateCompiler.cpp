// Source includes
#include "squareAllJIT.hpp"
#include "../../src/jitti/jitti.hpp"
#include "../../src/Array.hpp"
#include "../../src/ChaiBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

using SquareAllType = void (*)( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer > const &,
                                LvArray::ArrayView< int const, 1, 0, std::ptrdiff_t, LvArray::ChaiBuffer > const & );

using AddNType = SquareAllType;

void test( jitti::Function< SquareAllType > const & squareAll )
{
  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::ChaiBuffer > output( 100 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::ChaiBuffer > input( 100 );

  for ( std::ptrdiff_t i = 0; i < input.size(); ++i )
  { input[ i ] = i; }

  // Call the function.
  squareAll( output.toView(), input.toViewConst() );

  // Check the output.
  output.move( LvArray::MemorySpace::CPU );
  for ( std::ptrdiff_t i = 0; i < output.size(); ++i )
  { EXPECT_EQ( output[ i ], i * i ); }
}

TEST( compileTemplate, serial )
{
  jitti::CompilationInfo info = getCompilationInfo();

  info.templateParams = "RAJA::loop_exec";
  
  info.outputObject = JITTI_OUTPUT_DIR "/squareAllJITSerial.o";
  info.outputLibrary = JITTI_OUTPUT_DIR "/libsquareAllJITSerial.so";

  jitti::Function< SquareAllType > const squareAll = jitti::compileTemplate< SquareAllType >( info );

  test( squareAll );
}

#if defined( RAJA_ENABLE_OPENMP )
TEST( compileTemplate, OpenMP )
{
  jitti::CompilationInfo info = getCompilationInfo();

  info.templateParams = "RAJA::omp_parallel_for_exec";
  
  info.outputObject = JITTI_OUTPUT_DIR "/squareAllJITOpenMP.o";
  info.outputLibrary = JITTI_OUTPUT_DIR "/libsquareAllJITOpenMP.so";

  jitti::Function< SquareAllType > const squareAll = jitti::compileTemplate< SquareAllType >( info );

  test( squareAll );
}
#endif

#if defined( LVARRAY_USE_CUDA )
TEST( compileTemplate, CUDA )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::cuda_exec< 32 >";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/squareAllJITCUDA.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllJITCUDA.so";

  jitti::Function< SquareAllType > const squareAll = jitti::compileTemplate< SquareAllType >( info.compileCommand,
                                                                                              compilerIsNVCC,
                                                                                              info.linker,
                                                                                              info.linkArgs,
                                                                                              info.function,
                                                                                              templateParams,
                                                                                              info.header,
                                                                                              outputObject,
                                                                                              outputLib );

  test( squareAll );
}
#endif

TEST( TypedCache, serial )
{
  jitti::CompilationInfo info = getCompilationInfo();
  jitti::TypedCache< SquareAllType > cache( info.compilationTime, JITTI_OUTPUT_DIR );

  info.templateParams = "RAJA::loop_exec";
  
  jitti::Function< SquareAllType > const & squareAll = cache.getOrLoadOrCompile( info );

  test( squareAll );
}

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

