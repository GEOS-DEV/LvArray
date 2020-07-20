// Source includes
#include "jitti/jitti.hpp"
#include "squareAll.hpp"
#include "Array.hpp"
#include "MallocBuffer.hpp"

#include "jittiConfig.hpp"

// TPL includes
#include <gtest/gtest.h>


TEST( JIT, SquareAll )
{
  jitti::CompilationInfo const info = getCompilationInfo();

  // The compiler to use and the standard compilation flags. These would be set by CMake in a header file.
  jitti::TemplateCompiler compiler( info.compilerPath, info.compilerFlags );

  // The function to compile.
  std::string const function = "squareAll";
  std::string const templateParams = "RAJA::omp_parallel_for_exec";

  // Compile the source and load it as a dynamic library.
  std::string const outputLib = "/usr/WS2/corbett5/LvArray/build-quartz-clang@10.0.0-debug/lib/libsquareAll.so";
  jitti::TypedDynamicLibrary dl = compiler.instantiateTemplate( function,
                                                                templateParams,
                                                                info.header,
                                                                outputLib,
                                                                info.includeDirs,
                                                                info.systemIncludeDirs,
                                                                info.libs );

  // Load the function from the library.
  using SquareAllType = void (*)( LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const,
                                  LvArray::ArraySlice< int, 1, 0, std::ptrdiff_t > const );

  std::string const name = "squareAll< " + templateParams + " >";
  SquareAllType const squareAll = dl.getSymbol< SquareAllType >( name.c_str() );

  // Prepare the input.
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > output( 10 );
  LvArray::Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > input( 10 );

  for ( std::ptrdiff_t i = 0; i < input.size(); ++i )
  { input[ i ] = i; }

  // Call the function.
  squareAll( output, input );

  // Check the output.
  for ( std::ptrdiff_t i = 0; i < output.size(); ++i )
  { EXPECT_EQ( output[ i ], i * i ); }
}

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}

