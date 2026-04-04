// Source includes
#include "jittiUnitTestConfig.hpp"
#include "jitti/Function.hpp"

#include "Array.hpp"
#include "ChaiBuffer.hpp"

#include "simpleTemplates.hpp"
#include "moreComplicatedTemplates.hpp"
#include "fullOnTemplates.hpp"
#include "testFunctionCompileCommands.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace jitti
{
namespace testing
{

#if defined(LVARRAY_USE_CUDA)
  constexpr bool compilerIsNVCC = true;
#else
  constexpr bool compilerIsNVCC = false;
#endif

TEST( simpleTemplates, add )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "add";
  info.templateParams = "5";
  info.headerFile = simpleTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_add_5.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_add_5.so";

  // Compile, load, test and then unload add5.
  {
    jitti::Function< int (*)( int ) > const add5( info );
    EXPECT_EQ( add5( 0 ), add< 5 >( 0 ) );
    EXPECT_EQ( add5( 5 ), add< 5 >( 5 ) );
  }

  // Load add5 again.
  jitti::Function< int (*)( int ) > const add5( info.outputLibrary );
  EXPECT_EQ( add5( 10 ), add< 5 >( 10 ) );
  EXPECT_EQ( add5( 15 ), add< 5 >( 15 ) );

  info.templateParams = "1024";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_add_1024.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_add_1024.so";

  // Compile, load, test and then unload add1024.
  {
    jitti::Function< int (*)( int ) > const add1024( info );
    EXPECT_EQ( add1024( 0 ), add< 1024 >( 0 ) );
    EXPECT_EQ( add1024( 1024 ), add< 1024 >( 1024 ) );
  }

  // Load add1024 again.
  jitti::Function< int (*)( int ) > const add1024( info.outputLibrary );
  EXPECT_EQ( add1024( 2048 ), add< 1024 >( 2048 ) );
  EXPECT_EQ( add1024( 3072 ), add< 1024 >( 3072 ) );

  // Make sure add5 still works.
  EXPECT_EQ( add5( 20 ), add< 5 >( 20 ) );
  EXPECT_EQ( add5( 25 ), add< 5 >( 25 ) );
}

TEST( simpleTemplates, squareAll )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "squareAll";
  info.headerFile = simpleTemplatesPath;

  // Prepare to compile squareAll< int >
  info.templateParams = "int";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_squareAll_int.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_squareAll_int.so";

  std::vector< int > const intInput { 0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10 };
  std::vector< int > intOutput;
  
  std::vector< int > intExpectedOutput( intInput.size() );
  squareAll( intExpectedOutput.data(), intInput.data(), intExpectedOutput.size() );

  // Compile, load, test and then unload squareAll< int >.
  {
    jitti::Function< void (*)( int *, int const *, int ) > const squareAllInt( info );
    intOutput = intInput;
    squareAllInt( intOutput.data(), intInput.data(), intOutput.size() );
    EXPECT_EQ( intOutput, intExpectedOutput );
  }

  // Load squareAll< int > again.
  jitti::Function< void (*)( int *, int const *, int ) > const squareAllInt( info.outputLibrary );
  intOutput = intInput;
  squareAllInt( intOutput.data(), intInput.data(), intOutput.size() );
  EXPECT_EQ( intOutput, intExpectedOutput );

  // Prepare to compile squareAll< float >
  info.templateParams = "float";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_squareAll_float.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_squareAll_float.so";

  std::vector< float > const floatInput { 0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10 };
  std::vector< float > floatOutput;
  
  std::vector< float > floatExpectedOutput( floatInput.size() );
  squareAll( floatExpectedOutput.data(), floatInput.data(), floatExpectedOutput.size() );

  // Compile, load, test and then unload squareAllFloat.
  {
    jitti::Function< void (*)( float *, float const *, int ) > const squareAllFloat( info );
    floatOutput = floatInput;
    squareAllFloat( floatOutput.data(), floatInput.data(), floatOutput.size() );
    EXPECT_EQ( floatOutput, floatExpectedOutput );
  }

  // Load squareAll< float > again.
  jitti::Function< void (*)( float *, float const *, int ) > const squareAllFloat( info.outputLibrary );
  floatOutput = floatInput;
  squareAllFloat( floatOutput.data(), floatInput.data(), floatOutput.size() );
  EXPECT_EQ( floatOutput, floatExpectedOutput );

  // Make sure squareAll< int > still works.
  intOutput = intInput;
  squareAllInt( intOutput.data(), intInput.data(), intOutput.size() );
  EXPECT_EQ( intOutput, intExpectedOutput );
}

TEST( moreComplicatedTemplates, addToString )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "addToString";
  info.templateParams = "5";
  info.headerFile = moreComplicatedTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_addToString_5.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_addToString_5.so";

  // Compile, load, and test addToString5.
  jitti::Function< std::string (*)( std::string const & ) > const addToString5( info );
  EXPECT_EQ( addToString5( "foo" ), addToString< 5 >( "foo" ) );
  EXPECT_EQ( addToString5( "bar" ), addToString< 5 >( "bar" ) );
}

TEST( moreComplicatedTemplates, staticMapAccess )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "staticMapAccess";
  info.templateParams = "std::string, int";
  info.headerFile = moreComplicatedTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_staticMapAccess_std::string_int.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_staticMapAccess_std::string_int.so";

  // Populate the map as { "five": 5, "size": 6 }
  staticMapAccess< std::string, int >( "five", 5 );
  staticMapAccess< std::string, int >( "six", 6 );

  // Verify that the JIT'ed function sees the same map.
  jitti::Function< int& (*)( std::string const &, int const & ) > const staticMapAccessStringInt( info );
  EXPECT_EQ( staticMapAccessStringInt( "five", 0 ), 5 );
  EXPECT_EQ( staticMapAccessStringInt( "six", 0 ), 6 );

  // Update the map via the JIT'ed function to { "five": -1, "six": 6, "seven", 7 }
  staticMapAccessStringInt( "five", 0 ) = -1;
  staticMapAccessStringInt( "seven", 7 );

  EXPECT_EQ( ( staticMapAccess< std::string, int >( "five", 0 ) ), -1 );
  EXPECT_EQ( ( staticMapAccess< std::string, int >( "six", 0 ) ), 6 );
  EXPECT_EQ( ( staticMapAccess< std::string, int >( "seven", 0 ) ), 7 );
}

TEST( moreComplicatedTemplates, factory )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "factory";
  info.templateParams = "int";
  info.headerFile = moreComplicatedTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_factory_int.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_factory_int.so";

  // Compile, load, and test factoryInt.
  jitti::Function< std::unique_ptr< Base > (*)( int const & ) > const factoryInt( info );
  EXPECT_EQ( factoryInt( 5 )->getValueString(), "5" );
  EXPECT_EQ( LvArray::system::demangleType( *factoryInt( 5 ) ), "Derived<int>" );

  info.templateParams = "double";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_factory_double.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_factory_double.so";

  // Compile, load, and test factoryDouble.
  jitti::Function< std::unique_ptr< Base > (*)( double const & ) > const factoryDouble( info );
  EXPECT_EQ( factoryDouble( 3.14 )->getValueString(), "3.14" );
  EXPECT_EQ( LvArray::system::demangleType( *factoryDouble( 3.14 ) ), "Derived<double>" );
}

TEST( generatedCommands, add )
{
  jitti::CompilationInfo info;

  info.compileCommand = testFunction_COMPILE_COMMAND;
  info.compilerIsNVCC = compilerIsNVCC;
  info.linker = testFunction_LINKER;
  info.linkArgs = testFunction_LINK_ARGS;
  info.templateFunction = "add";
  info.templateParams = "7";
  info.headerFile = simpleTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_generated_add7.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_generated_add7.so";

  // Compile, load, and test add< 7 >.
  jitti::Function< int (*)( int ) > const add7( info );
  EXPECT_EQ( add7( 5 ), add< 7 >( 5 ) );
}

TEST( fullOnTemplates, getUmpireAllocatorName )
{
  jitti::CompilationInfo info;

  info.compileCommand = testFunction_COMPILE_COMMAND;
  info.compilerIsNVCC = compilerIsNVCC;
  info.linker = testFunction_LINKER;
  info.linkArgs = testFunction_LINK_ARGS;
  info.templateFunction = "getUmpireAllocatorName";
  info.templateParams = "int";
  info.headerFile = fullOnTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_getUmpireAllocatorName_int.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_getUmpireAllocatorName_int.so";

  // Compile, load, and test add< 7 >.
  jitti::Function< std::string (*)( int * ) > const getUmpireAllocatorNameInt( info );
  LvArray::Array< int, 1, RAJA::PERM_I, int, LvArray::ChaiBuffer > array( 10 );
  EXPECT_EQ( getUmpireAllocatorNameInt( array.data() ), "HOST" );

#if defined(LVARRAY_USE_CUDA)
  array.move( LvArray::MemorySpace::GPU );
  EXPECT_EQ( getUmpireAllocatorNameInt( array.data() ), "DEVICE" );
#endif
}

TEST( fullOnTemplates, cubeAll )
{
  using FuncType = void (*)( LvArray::ArrayView< int, 1, 0, int, LvArray::ChaiBuffer > const &,
                             LvArray::ArrayView< int const, 1, 0, int, LvArray::ChaiBuffer > const & );

  jitti::CompilationInfo info;

  info.compileCommand = testFunction_COMPILE_COMMAND;
  info.compilerIsNVCC = compilerIsNVCC;
  info.linker = testFunction_LINKER;
  info.linkArgs = testFunction_LINK_ARGS;
  info.templateFunction = "cubeAll";
  info.templateParams = "RAJA::loop_exec";
  info.headerFile = fullOnTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_fullOnTemplates_cubeAll_serial.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_fullOnTemplates_cubeAll_serial.so";

  LvArray::Array< int, 1, RAJA::PERM_I, int, LvArray::ChaiBuffer > src( 10 );
  LvArray::Array< int, 1, RAJA::PERM_I, int, LvArray::ChaiBuffer > dst( src );

  for ( int i = 0; i < src.size(); ++i )
  {
    src[ i ] = i;
  }

  jitti::Function< FuncType > const cubeAll_loop_exec( info );
  cubeAll_loop_exec( src, dst );

  for( int i = 0; i < src.size(); ++i )
  {
    EXPECT_EQ( dst[ i ], src[ i ] * src[ i ] * src[ i ] );
  }

  dst.setValues< RAJA::loop_exec >( 0 );

#if defined( RAJA_ENABLE_OPENMP )
  info.templateParams = "RAJA::omp_parallel_for_exec";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_fullOnTemplates_cubeAll_omp.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_fullOnTemplates_cubeAll_omp.so";
  jitti::Function< FuncType > const cubeAll_omp( info );
  cubeAll_omp( dst, src );

  for( int i = 0; i < src.size(); ++i )
  {
    EXPECT_EQ( dst[ i ], src[ i ] * src[ i ] * src[ i ] );
  }

  dst.setValues< RAJA::loop_exec >( 0 );
#endif

#if defined( LVARRAY_USE_CUDA)
  info.templateParams = "RAJA::cuda_exec< 32 >";
  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_fullOnTemplates_cubeAll_cuda.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_fullOnTemplates_cubeAll_cuda.so";
  jitti::Function< FuncType > const cubeAll_cuda( info );
  cubeAll_cuda( dst, src );

  dst.move( LvArray::MemorySpace::CPU );
  for( int i = 0; i < src.size(); ++i )
  {
    EXPECT_EQ( dst[ i ], src[ i ] * src[ i ] * src[ i ] );
  }
#endif
}

TEST( Function, deathTest )
{
  jitti::CompilationInfo info;

  info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
  info.compilerIsNVCC = false;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "add";
  info.templateParams = "5";
  info.headerFile = simpleTemplatesPath;

  info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_add_5_deathTest.o";
  info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_add_5_deathTest.so";

  jitti::Function< int (*)( int ) > const add5( info );

  EXPECT_DEATH_IF_SUPPORTED( jitti::Function< double (*)( int ) >( info.outputLibrary ), "" );
  EXPECT_DEATH_IF_SUPPORTED( jitti::Function< int (*)( int, int ) >( info.outputLibrary ), "" );
}

} // namespace testing
} // namespace jitti

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
