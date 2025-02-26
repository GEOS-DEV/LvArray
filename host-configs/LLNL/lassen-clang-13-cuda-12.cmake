set(CONFIG_NAME "lassen-clang-13-cuda-12" CACHE PATH "")

#set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-17.0.6/ ) # Fails at chai-2023.06.0 compilation
#set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-16.0.6-gcc-11.2.1/ ) # Fails at "-- CUDA Support is ON"
#set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-16.0.6-gcc-8.3.1/ ) # Fails at "-- CUDA Support is ON"
#set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-16.0.6/ ) # Fails at RAJA-v2023.06.1 compilation
#set(COMPILER_DIR /usr/tce/packages/clang/clang-16.0.6-gcc-11.2.1/ ) # Fails at "-- CUDA Support is ON"
#set(COMPILER_DIR /usr/tce/packages/clang/clang-16.0.6-gcc-8.3.1/ ) # Fails at "-- CUDA Support is ON"
#set(COMPILER_DIR /usr/tce/packages/clang/clang-ibm-15.0.6/ ) # Fails at chai-2023.06.0 compilation
#set(COMPILER_DIR /usr/tce/packages/clang/clang-15.0.6/ ) # Fails at chai-2023.06.0 compilation

# Set clang info
set(COMPILER_DIR /usr/tce/packages/clang/clang-13.0.1-gcc-8.3.1/ )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/clang CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/clang++ CACHE PATH "")

# Set a few clang binaries explicitly, otherwise CMake might find them in another path
set(CLANGFORMAT_EXECUTABLE ${COMPILER_DIR}/bin/clang-format CACHE PATH "")
set(CLANGQUERY_EXECUTABLE ${COMPILER_DIR}/bin/clang-query CACHE PATH "")
set(CLANGTIDY_EXECUTABLE ${COMPILER_DIR}/bin/clang-tidy CACHE PATH "")

# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -mcpu=powerpc64le -mtune=powerpc64le" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/lassen-cuda-12-base.cmake)
