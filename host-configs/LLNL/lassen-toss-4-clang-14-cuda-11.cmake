set(CONFIG_NAME "lassen-toss-4-clang-14-cuda-11" CACHE PATH "")

set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-14.0.6)
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/clang CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/clang++ CACHE PATH "")

# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/lassen-cuda-11-base.cmake)