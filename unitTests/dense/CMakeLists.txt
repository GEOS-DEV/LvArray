###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

#
# Specify list of tests
#
set( testSources
     testEigenDecomposition.cpp
     testLinearSolve.cpp
   )

#
# Add gtest C++ based tests
#
foreach(test ${testSources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest lvarray lvarraydense ${lvarray_dependencies} )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

    install(TARGETS ${test_name}
            DESTINATION bin)
endforeach()


