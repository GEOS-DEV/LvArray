#include "squareAll.hpp"


#if defined(__CUDACC__)


std::string const cudaCompilationCommand = "/usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc -ccbin=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.26/bin/mpicxx  -Iinclude -isystem=/usr/tce/packages/cuda/cuda-10.1.243/include -isystem=/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2019.06.24/include -isystem=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.26/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/raja/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/caliper/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/chai/include  -restrict -arch sm_70 --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations -g -G -O0 -Xcompiler -O0 -Xcompiler=-fPIC   -Xcompiler=-fopenmp=libomp -std=c++14 -x cu -c /usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.cpp -o unitTests/jitti/CMakeFiles/squareAll.dir/squareAll.cpp.o && /usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc -ccbin=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.26/bin/mpicxx  -Iinclude -isystem=/usr/tce/packages/cuda/cuda-10.1.243/include -isystem=/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-2019.06.24/include -isystem=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-clang-upstream-2019.03.26/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/raja/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/caliper/include -isystem=/usr/gapps/GEOSX/thirdPartyLibs/2020-07-08/install-lassen-clang@upstream-release/chai/include  -restrict -arch sm_70 --expt-extended-lambda -Werror cross-execution-space-call,reorder,deprecated-declarations -g -G -O0 -Xcompiler -O0 -Xcompiler=-fPIC   -Xcompiler=-fopenmp=libomp -std=c++14 -x cu -M /usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.cpp -MT unitTests/jitti/CMakeFiles/squareAll.dir/squareAll.cpp.o -o $DEP_FILE";

std::string getCompilationCommand()
{
  return cudaCompilationCommand.substr( 0, cudaCompilationCommand.find("-o") ); 
}

#else

#include "../../src/jitti/json.hpp"

std::string getCompilationCommand()
{ 
  std::ifstream compileCommands( JITTI_COMPILE_COMMANDS_JSON );
  nlohmann::json j;
  compileCommands >> j;

  // Iterate over all the compilation commands and get the compilation command for the current translation unit.
  for ( auto const & translationUnit : j )
  {
    if ( translationUnit[ "file" ] == "/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.cpp" )
    {
      std::string const command = translationUnit[ "command" ].get<std::string>();
      return command.substr( 0, command.find("-o") ); 
    }
  }

  LVARRAY_ERROR( "Could not find the compilation command for " __FILE__ " in " JITTI_COMPILE_COMMANDS_JSON );
  return "";
}

#endif


jitti::CompilationInfo getCompilationInfo()
{
  jitti::CompilationInfo info;

  // info.compilerPath = JITTI_COMPILER_PATH;
  info.compilerPath = getCompilationCommand();
  
  // info.compilerFlags = "-Wall -Wextra -Werror -Wpedantic -pedantic-errors -Wshadow -Wfloat-equal "
  //                      "-Wcast-align -Wcast-qual -fcolor-diagnostics -g -fstandalone-debug "
  //                      "-fopenmp=libomp -std=c++14";
  
  info.header = "/usr/WS2/corbett5/LvArray/unitTests/jitti/squareAll.hpp";
  
  // info.includeDirs = {
  //   "/usr/WS2/corbett5/LvArray/src",
  //   "/usr/WS2/corbett5/LvArray/build-quartz-clang@10.0.0-debug/include"
  // };

  // info.systemIncludeDirs = {
  //   "/usr/gapps/GEOSX/thirdPartyLibs/2020-06-17/install-quartz-clang@10.0.0-release/raja/include"
  // };

  info.libs = {
    JITTI_RAJA_LIBRARY
  };

  return info;
}
