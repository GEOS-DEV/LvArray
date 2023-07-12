# Set up the tpls
include(${CMAKE_CURRENT_LIST_DIR}/llnl-tpls-base.cmake)

set(CAMP_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")
set(RAJA_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")
set(UMPIRE_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CHAI ON CACHE BOOL "")
set(CHAI_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CALIPER ON CACHE BOOL "")
set(ENABLE_ADIAK ON CACHE BOOL "" )
set(CALIPER_DIR ${GEOSX_TPL_DIR}/caliper CACHE PATH "")

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

# Uncomment this line to make nvcc output register usage for each kernel.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage" CACHE STRING "" FORCE)

# GTEST options
set(ENABLE_GTEST_DEATH_TESTS OFF CACHE BOOL "")
set(gtest_disable_pthreads ON CACHE BOOL "")

# Documentation
set(ENABLE_UNCRUSTIFY OFF CACHE BOOL "" FORCE)
set(ENABLE_DOXYGEN OFF CACHE BOOL "" FORCE)