#pragma once

#include <string>
#include <unordered_map>

namespace jitti
{
namespace utils
{

/**
 * 
 */
std::string compileTemplate( std::string const & compileCommand,
                             bool const compilerIsNVCC,
                             std::string const & linker,
                             std::string const & linkArgs,
                             std::string const & templateFunction,
                             std::string const & templateParams,
                             std::string const & headerFile,
                             std::string const & outputObject,
                             std::string const & outputLibrary );

void makeDirsForPath( std::string const & path );

/**
 * 
 */
void readDirectoryFiles( time_t const time,
                         std::string const & directory,
                         std::unordered_map< std::string, std::string > & libraries );

} // namespace utils
} // namespace jitti
