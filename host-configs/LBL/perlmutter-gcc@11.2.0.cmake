set(CONFIG_NAME "perlmutter-gcc@11.2.0" CACHE PATH "") 

set(COMPILER_DIR  /opt/cray/pe/craype/2.7.20)
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/cc CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/CC CACHE PATH "")

# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -target-accel=nvidia80 " CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -target-accel=nvidia80  ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -target-accel=nvidia80 -g " CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/perlmutter-base.cmake)
