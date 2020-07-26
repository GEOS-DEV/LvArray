#include "squareAllJIT.hpp"
#include "squareAllCompileCommands.hpp"

jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  info.compileCommand = squareAllJIT_COMPILE_COMMAND;
  info.linker = squareAllJIT_LINKER;
  info.linkArgs = squareAllJIT_LINK_ARGS;
  info.header = "/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAllJIT.hpp";
  info.function = "squareAll";

  return info;
}
