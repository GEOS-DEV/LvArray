// Source includes
#include "testHelpers.hpp"
#include "squareAllJIT.hpp"
#include "../../src/jitti/jitti.hpp"

// TPL includes
#include <gtest/gtest.h>

TEST( TemplateCompiler, serial )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::loop_exec";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/squareAllJITSerial.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllJITSerial.so";

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

  test( squareAll, "RAJA::policy::loop::loop_exec" );
}

#if defined( USE_OPENMP )
TEST( TemplateCompiler, OpenMP )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::omp_parallel_for_exec";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/squareAllJITOpenMP.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllJITOpenMP.so";

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

  test( squareAll, "RAJA::policy::omp::omp_parallel_for_exec" );
}
#endif

#if defined( USE_CUDA )
TEST( TemplateCompiler, CUDA )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  std::string const templateParams = "RAJA::cuda_exec< 32 >";
  
  std::string const outputObject = JITTI_OUTPUT_DIR "/squareAllJITCUDA.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libsquareAllJITCUDA.so";

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

  test( squareAll, "RAJA::policy::cuda::cuda_exec<32ul, false>" );
}
#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

