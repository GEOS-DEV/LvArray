#include "jittiConfig.hpp"
#include "../../src/jitti/CompilationInfo.hpp"
#include "../../src/Macros.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <ctime>

namespace jitti
{
namespace testing
{

/**
 * Test that the @c CompilationInfo constructor initializes the @c compilationTime field
 * to the time that the translation unit it is constructed in was compiled.
 */
TEST( CompilationInfo, Constructor )
{
  CompilationInfo info;
  EXPECT_EQ( info.compilationTime, internal::getCompileTime( __DATE__, __TIME__ ) );
  EXPECT_GE( time( nullptr ), info.compilationTime );

  char const * const timeString = ctime( &info.compilationTime );
  EXPECT_EQ( std::string( timeString + 4, 6 ), std::string( __DATE__, 6 ) );
  EXPECT_EQ( std::string( timeString + 11, 8 ), std::string( __TIME__ ) );
  EXPECT_EQ( std::string( timeString + 20, 4 ), std::string( &__DATE__[ 7 ] ) );
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

