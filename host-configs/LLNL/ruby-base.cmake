set(ENABLE_FORTRAN OFF CACHE BOOL "")

include(${CMAKE_CURRENT_LIST_DIR}/llnl-tpls-base.cmake)

set(CAMP_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")
set(RAJA_DIR ${GEOSX_TPL_DIR}/raja CACHE PATH "")

set(ENABLE_UMPIRE ON CACHE BOOL "")
set(UMPIRE_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CHAI ON CACHE BOOL "")
set(CHAI_DIR ${GEOSX_TPL_DIR}/chai CACHE PATH "")

set(ENABLE_CALIPER ON CACHE BOOL "")
set(CALIPER_DIR ${GEOSX_TPL_DIR}/caliper CACHE PATH "")

# set(ENABLE_PYLVARRAY ON CACHE BOOL "")
# set(PYTHON_DIR /usr/tce/packages/python/python-3.7.2 CACHE PATH "")

set(SPHINX_EXECUTABLE /collab/usr/gapps/python/build/spack-toss3.2/opt/spack/linux-rhel7-x86_64/gcc-4.9.3/python-2.7.14-7rci3jkmuht2uiwp433afigveuf4ocnu/bin/sphinx-build CACHE PATH "")

set(DOXYGEN_EXECUTABLE ${GEOSX_TPL_DIR}/doxygen/bin/doxygen CACHE PATH "")

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "")

set(ENABLE_ADDR2LINE ON CACHE BOOL "")

set(ENABLE_CUDA OFF CACHE BOOL "")

set(ENABLE_TOTALVIEW_OUTPUT OFF CACHE BOOL "Enables Totalview custom view" FORCE)
