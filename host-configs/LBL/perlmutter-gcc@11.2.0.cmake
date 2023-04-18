set(CONFIG_NAME "perlmutter-gcc@11.2.0" CACHE PATH "") 

#set(COMPILER_DIR  /opt/cray/pe/craype/2.7.19)
#set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/gcc CACHE PATH "")
#set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/g++ CACHE PATH "")

# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -target-accel=nvidia80 -DRAJA_ENABLE_VECTORIZATION=OFF" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -target-accel=nvidia80 -DRAJA_ENABLE_VECTORIZATION=OFF ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -target-accel=nvidia80 -g -DRAJA_ENABLE_VECTORIZATION=OFF" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/perlmutter-base.cmake)
