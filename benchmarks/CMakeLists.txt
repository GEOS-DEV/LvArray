#
# Specify list of tests
#
set(benchmarkSources
    benchmarkReduce.cpp
    benchmarkInnerProduct.cpp
    benchmarkOuterProduct.cpp
    benchmarkMatrixVector.cpp
    benchmarkMatrixMatrix.cpp
    benchmarkArray1DR2TensorMultiplication.cpp
   )

if (NOT ${ENABLE_BENCHMARKS})
    message(FATAL_ERROR "Benchmarks not enabled!")
endif()

#
# Add gtest C++ based tests
#
foreach(benchmark ${benchmarkSources})
    get_filename_component( benchmark_name ${benchmark} NAME_WE )
    blt_add_executable( NAME ${benchmark_name}
                        SOURCES ${benchmark} ${benchmark_name}Kernels.cpp
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gbenchmark gtest lvarray ${lvarray_dependencies} )

    target_include_directories( ${benchmark_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )
    
    # blt_add_target_compile_flags( TO ${benchmark_name}
    #                               FLAGS -fsave-optimization-record )

    blt_add_benchmark( NAME ${benchmark_name}
                       COMMAND ${benchmark_name}
                     )

endforeach()
