set(CONFIG_NAME "lassen-clang@upstream" CACHE PATH "") 


set(COMPILER_DIR /usr/tce/packages/clang/clang-upstream-2019.03.26 )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/clang CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/clang++ CACHE PATH "")



# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=/usr/tce/packages/gcc/gcc-8.3.1" CACHE STRING "")


include(${CMAKE_CURRENT_LIST_DIR}/lassen-base.cmake)
