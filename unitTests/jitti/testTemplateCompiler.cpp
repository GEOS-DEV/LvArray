// Source includes
#include "../../src/jitti/jitti.hpp"
#include "../../src/Array.hpp"
#include "../../src/MallocBuffer.hpp"

#include "squareAll.hpp"

#include "jittiConfig.hpp"

// TPL includes
#include <gtest/gtest.h>

// Load the function from the library.
using SquareAllType = void (*)( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const,
                                LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const );

void test( SquareAllType squareAll )
{
  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > output( 100 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > input( 100 );

  for ( std::ptrdiff_t i = 0; i < input.size(); ++i )
  { input[ i ] = i; }

  // Call the function.
  squareAll( output, input );

  // Check the output.
  for ( std::ptrdiff_t i = 0; i < output.size(); ++i )
  { EXPECT_EQ( output[ i ], i * i ); }
}

TEST( TemplateCompiler, serial )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::loop_exec";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/libsquareAllSerial.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllSerial.so";

  // The compiler to use and the standard compilation flags. These would be set by CMake in a header file.
  jitti::TemplateCompiler compiler( info.compileCommand, info.linker, info.linkArgs );

  // Compile the source and load it as a dynamic library.
  jitti::TypedDynamicLibrary dl = compiler.instantiateTemplate( info.function,
                                                                templateParams,
                                                                info.header,
                                                                outputObject,
                                                                outputLib );

  std::string const name = info.function + "< " + templateParams + " >";
  SquareAllType const squareAll = dl.getSymbol< SquareAllType >( name.c_str() );

  test( squareAll );
}

#if defined( USE_OPENMP )
TEST( TemplateCompiler, OpenMP )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::omp_parallel_for_exec";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/libsquareAllOpenMP.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllOpenMP.so";

  // The compiler to use and the standard compilation flags. These would be set by CMake in a header file.
  jitti::TemplateCompiler compiler( info.compileCommand, info.linker, info.linkArgs );

  // Compile the source and load it as a dynamic library.
  jitti::TypedDynamicLibrary dl = compiler.instantiateTemplate( info.function,
                                                                templateParams,
                                                                info.header,
                                                                outputObject,
                                                                outputLib );

  // Load the function from the library.
  std::string const name = info.function + "< " + templateParams + " >";
  SquareAllType const squareAll = dl.getSymbol< SquareAllType >( name.c_str() );

  test( squareAll );
}
#endif

#if defined( USE_CUDA )
TEST( TemplateCompiler, CUDA )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::cuda_exec< 32 >";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/libsquareAllCUDA.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllCUDA.so";

  // The compiler to use and the standard compilation flags. These would be set by CMake in a header file.
  jitti::TemplateCompiler compiler( info.compileCommand, info.linker, info.linkArgs );

  // Compile the source and load it as a dynamic library.
  jitti::TypedDynamicLibrary dl = compiler.instantiateTemplate( info.function,
                                                                templateParams,
                                                                info.header,
                                                                outputObject,
                                                                outputLib );

  // Load the function from the library.
  std::string const name = info.function + "< " + templateParams + " >";
  SquareAllType const squareAll = dl.getSymbol< SquareAllType >( name.c_str() );

  test( squareAll );
}
#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

