#include "squareAllJIT.hpp"
#include "squareAllCompileCommands.hpp"
#include "squareAll.hpp"

jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  info.compileCommand = squareAllJIT_COMPILE_COMMAND;
  info.linker = squareAllJIT_LINKER;
  info.linkArgs = squareAllJIT_LINK_ARGS;

  std::string const currentFile = __FILE__;
  info.header = currentFile.substr( 0, currentFile.size() - ( sizeof( "squareAllJIT.cpp" ) - 1 ) )
                                 + "squareAll.hpp";

  info.function = "squareAll";

  return info;
}
