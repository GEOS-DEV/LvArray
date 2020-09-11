set(CONFIG_NAME "quartz-tribol2-clang@10.0.0" CACHE PATH "")

set(COMPILER_DIR /usr/tce/packages/clang/clang-10.0.0/ )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/clang CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/clang++ CACHE PATH "")

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native" CACHE STRING "")

set(RAJA_DIR /usr/WS2/corbett5/CHAI3/install-quartz-tribol2-clang@10.0.0-release CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")
set(UMPIRE_DIR /usr/WS2/corbett5/CHAI3/install-quartz-tribol2-clang@10.0.0-release CACHE PATH "")

set(ENABLE_CHAI ON CACHE BOOL "")
set(CHAI_DIR /usr/WS2/corbett5/CHAI3/install-quartz-tribol2-clang@10.0.0-release CACHE PATH "")

set(ENABLE_CALIPER OFF CACHE BOOL "")
set(ENABLE_PAPI OFF CACHE BOOL "")
set(USE_ADDR2LINE ON CACHE BOOL "")

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")

set(ENABLE_OPENMP ON CACHE BOOL "")
set(CUDA_ENABLED OFF CACHE BOOL "")

set(ENABLE_TOTALVIEW_OUTPUT OFF CACHE BOOL "Enables Totalview custom view" FORCE)
