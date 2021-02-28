#include "jittiConfig.hpp"
#include "../../src/jitti/utils.hpp"
#include "../../src/Macros.hpp"

// TPL includes
#include <gtest/gtest.h>

#include <stdio.h> // for remvoe
#include <unistd.h> // for access

namespace jitti
{
namespace testing
{

/**
 * @brief Return the current working directory.
 * @return The current working directory.
 */
static std::string getCurrentDirectory()
{
  char * const currentDirectory = get_current_dir_name();
  std::string const ret( currentDirectory );
  free( currentDirectory );
  return ret;
}

/**
 * @brief Create a file at the given path.
 * @param path The path of the file to create.
 */
static void createFile( std::string const & path )
{
  FILE * const fp = fopen( path.c_str(), "w" );
  EXPECT_NE( fp, nullptr ) << "Could not fopen " << path.c_str();

  if( fp != nullptr )
  {
    EXPECT_EQ( fclose( fp ), 0 );
  }
}

static void checkReadDirectoryFiles( time_t const t,
                                     std::string const & dirName,
                                     std::unordered_map< std::string, std::string > librariesInDir,
                                     std::vector< std::string > const & expectedLibraries )
{
  for( int i : { 0, 1 } )
  {
    LVARRAY_UNUSED_VARIABLE( i );

    utils::readDirectoryFiles( t, dirName, librariesInDir );
    EXPECT_EQ( librariesInDir.size(), expectedLibraries.size() );

    for( std::string const & fileName : expectedLibraries )
    {
      std::string const filePath = dirName + "/" + fileName;
      bool const fileFound = librariesInDir.count( fileName ) == 1;
      EXPECT_TRUE( fileFound ) << filePath;

      if( fileFound )
      {
        EXPECT_EQ( librariesInDir.at( fileName ), filePath );
      }
    }
  }
}

/**
 * Tests that utils::compileTemplate produces an executable file.
 */
TEST( utils, compileTemplate )
{
  std::string const currentFile = __FILE__;
  std::string const headerFile = currentFile.substr( 0, currentFile.size() - ( sizeof( "testUtils.cpp" ) - 1 ) )
                                 + "simpleTemplates.hpp";

  std::string const outputObject = JITTI_OUTPUT_DIR "/testUtilsAdd.o";
  std::string const outputLib = JITTI_OUTPUT_DIR "/libtestUtilsAdd.so";
  remove( outputLib.c_str() );
  EXPECT_NE( access( outputLib.c_str(), R_OK | X_OK ), 0 ); 

  std::string const ret = utils::compileTemplate( JITTI_CXX_COMPILER "-std=c++14",
                                                   false,
                                                   JITTI_LINKER,
                                                   "",
                                                   "add",
                                                   "5",
                                                   headerFile,
                                                   outputObject,
                                                   outputLib );

  EXPECT_EQ( outputLib, ret );
  EXPECT_EQ( access( outputLib.c_str(), R_OK | X_OK ), 0 ); 
}

/**
 * Check that utils::readDirectoryFiles works.
 */
TEST( utils, readDirectoryFiles )
{
  std::string const dirName = getCurrentDirectory() + "/testUtils_readDirectoryFilesTemp";
  std::string const deleteDirCommand = "rm -rf " + dirName;
  EXPECT_EQ( std::system( deleteDirCommand.c_str() ), 0 );

  EXPECT_EQ( mkdir( dirName.c_str(), 0700 ), 0 );

  time_t const t0 = time( nullptr );

  for( std::string const fileName : { "foo.so", "foo.o", "bar.so" } )
  {
    createFile( dirName + "/" + fileName );
  }

  std::unordered_map< std::string, std::string > librariesInDir;

  checkReadDirectoryFiles( t0, dirName, librariesInDir, { "foo.so", "bar.so" } );

  // Sleep one second to ensure the new files have a different time stamp.
  sleep( 1 );

  time_t const t1 = time( nullptr );

  for( std::string const fileName : { "fooNewer.so", "fooNewer.o", "barNewer.so" } )
  {
    createFile( dirName + "/" + fileName );
  }

  checkReadDirectoryFiles( t0, dirName, librariesInDir, { "foo.so", "bar.so", "fooNewer.so", "barNewer.so" } );

  librariesInDir.clear();

  checkReadDirectoryFiles( t1, dirName, librariesInDir, { "fooNewer.so", "barNewer.so" } );
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
