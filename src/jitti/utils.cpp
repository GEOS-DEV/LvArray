#include "utils.hpp"
#include "../Macros.hpp"

#include <sstream>
#include <regex>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

namespace jitti
{
namespace utils
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string compileTemplate( std::string const & compileCommand,
                             bool const compilerIsNVCC,
                             std::string const & linker,
                             std::string const & linkArgs,
                             std::string const & templateFunction,
                             std::string const & templateParams,
                             std::string const & headerFile,
                             std::string const & outputObject,
                             std::string const & outputLibrary )
{
  std::string const currentFile = __FILE__;
  std::string const sourceFile = currentFile.substr( 0, currentFile.size() - ( sizeof( "jitti.cpp" ) - 1 ) )
                                 + "templateSource.cpp"; 

  std::ostringstream fullCompileCommand;
  fullCompileCommand << compileCommand;
  if( compilerIsNVCC )
  {
    fullCompileCommand << " -Xcompiler";
  }

  fullCompileCommand << " -fPIC -c " + sourceFile + " -o " + outputObject;

  std::ostringstream joinedDefines;
  joinedDefines << " -D JITTI_TEMPLATE_HEADER_FILE='\"" << headerFile << "\"'"
                << " -D JITTI_TEMPLATE_FUNCTION=" << templateFunction
                << " -D JITTI_TEMPLATE_PARAMS=\"" << templateParams << "\""
                << " -D JITTI_TEMPLATE_PARAMS_STRING='\"" << templateParams << "\"'";

  std::string const joinedDefinesString = joinedDefines.str();

  // You need to escape commas for NVCC.
  if( compilerIsNVCC )
  {
    fullCompileCommand << std::regex_replace( joinedDefinesString, std::regex( "," ), "\\," );
  }
  else
  {
    fullCompileCommand << joinedDefinesString;
  }

  std::string const fullCompileCommandString = fullCompileCommand.str();

  LVARRAY_LOG( "\nCompiling " << sourceFile );
  LVARRAY_LOG_VAR( fullCompileCommandString );
  LVARRAY_ERROR_IF( std::system( fullCompileCommandString.c_str() ) != 0, fullCompileCommandString );

  std::string linkCommand = linker + " -shared " + outputObject + " " + linkArgs + " -o " + outputLibrary;

  LVARRAY_LOG( "\nLinking " << outputObject );
  LVARRAY_LOG_VAR( linkCommand );
  LVARRAY_ERROR_IF( std::system( linkCommand.c_str() ) != 0, linkCommand );

  return outputLibrary;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readDirectoryFiles( time_t const time,
                         std::string const & directory,
                         std::unordered_map< std::string, std::string > & libraries )
{
  DIR * const dirp = opendir( directory.c_str() );
  LVARRAY_ERROR_IF( dirp == nullptr, "Could not open '" << directory << "' as a directory." );

  struct dirent * dp;
  while( ( dp = readdir( dirp ) ) != nullptr )
  {
    std::string const fileName = dp->d_name;
    if( fileName.size() > 3 && fileName.substr( fileName.size() - 3 ) == ".so" )
    {
      std::string const filePath = directory + "/" + fileName;

      struct stat fileInfo;
      LVARRAY_ERROR_IF( stat( filePath.c_str(), &fileInfo ) != 0, "Error stat'ing '" << filePath << "'" );

      if( fileInfo.st_mtime > time )
      {
        auto const result = libraries.emplace( std::make_pair( std::move( fileName ), std::move( filePath ) ) );
        if( result.second )
        {
          LVARRAY_ERROR_IF_NE_MSG( result.first->second, filePath,
                                   "Two libraries with the same name exist in different directories." );
        }
      }
    }
  }

  closedir( dirp );
}

} // namespace utils
} // namespace jitti
