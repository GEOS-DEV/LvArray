// Source includes
#include "jittiUnitTestConfig.hpp"
#include "jitti/Cache.hpp"

#include "simpleTemplates.hpp"
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
  info.compilerIsNVCC = compilerIsNVCC;
  info.linker = JITTI_LINKER;
  info.linkArgs = "";
  info.templateFunction = "add";
  info.templateParams = "5";
  info.headerFile = simpleTemplatesPath;

  {
    jitti::Cache< int (*)( int ) > cache( 0, JITTI_TEST_OUTPUT_DIR );
    EXPECT_EQ( cache.getOrLoadOrCompile( info )( 7 ), 12 );
    EXPECT_EQ( cache.get( "add< 5 >" )( 8 ), 13 );

    info.templateParams = "5";
    EXPECT_FALSE( cache.tryGetOrLoad( "add< 8 >" ) );
    EXPECT_EQ( ache.getOrLoadOrCompile( info )( 8 ), 16 );
  }

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
