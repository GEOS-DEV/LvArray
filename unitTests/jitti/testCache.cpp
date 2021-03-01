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

// TEST( simpleTemplates, add )
// {
//   jitti::CompilationInfo info;

//   info.compileCommand = JITTI_CXX_COMPILER " -std=c++14";
//   info.compilerIsNVCC = compilerIsNVCC;
//   info.linker = JITTI_LINKER;
//   info.linkArgs = "";
//   info.templateFunction = "add";
//   info.templateParams = "5";
//   info.headerFile = simpleTemplatesPath;

//   info.outputObject = JITTI_TEST_OUTPUT_DIR "/testFunction_add_5.o";
//   info.outputLibrary = JITTI_TEST_OUTPUT_DIR "/libtestFunction_add_5.so";
// }

} // namespace testing
} // namespace jitti

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
