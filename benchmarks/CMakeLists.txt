###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

#
# Specify list of benchmarks
#
set( benchmarkSources
     benchmarkReduce.cpp
     benchmarkInnerProduct.cpp
     benchmarkOuterProduct.cpp
     benchmarkMatrixVector.cpp
     benchmarkMatrixMatrix.cpp
     benchmarkArray1DR2TensorMultiplication.cpp
     benchmarkSparsityGeneration.cpp
     benchmarkArrayOfArraysReduce.cpp
     benchmarkArrayOfArraysNodeToElementMapConstruction.cpp
     benchmarkEigendecomposition.cpp
   )

if( NOT ${ENABLE_BENCHMARKS} )
    message(FATAL_ERROR "Benchmarks not enabled!")
endif()

#
# Add benchmarks
#
foreach( benchmark ${benchmarkSources} )
    get_filename_component( benchmark_name ${benchmark} NAME_WE )
    blt_add_executable( NAME ${benchmark_name}
                        SOURCES ${benchmark} ${benchmark_name}Kernels.cpp
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gbenchmark gtest lvarray ${lvarray_dependencies} )

    target_include_directories( ${benchmark_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )
    
    # blt_add_target_compile_flags( TO ${benchmark_name}
    #                               FLAGS -fsave-optimization-record )

    blt_add_benchmark( NAME ${benchmark_name}
                       COMMAND ${benchmark_name} )

    install( TARGETS ${benchmark_name}
             DESTINATION bin )

endforeach()
