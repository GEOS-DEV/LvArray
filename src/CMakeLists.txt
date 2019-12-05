set(cxx-utilities_headers
    templateHelpers.hpp
    Array.hpp
    ArraySlice.hpp
    ArrayUtilities.hpp
    ArrayView.hpp
    arrayManipulation.hpp
    sortedArrayManipulation.hpp
    sortedArrayManipulationHelpers.hpp
    arrayHelpers.hpp
    IntegerConversion.hpp
    Macros.hpp
    SetFPE.hpp
    SetSignalHandling.hpp
    SortedArrayView.hpp
    SortedArray.hpp
    StackBuffer.hpp
    stackTrace.hpp
    StringUtilities.hpp
    ArrayOfArraysView.hpp
    ArrayOfArrays.hpp
    ArrayOfSetsView.hpp
    ArrayOfSets.hpp
    SparsityPatternView.hpp
    SparsityPattern.hpp
    CRSMatrixView.hpp
    CRSMatrix.hpp
    totalview/tv_data_display.h
    Permutation.hpp
    bufferManipulation.hpp
    ChaiBuffer.hpp
    MallocBuffer.hpp
   )

set(cxx-utilities_sources
    SetFPE.cpp
    SetSignalHandling.cpp
    stackTrace.cpp
    StringUtilities.cpp
    totalview/tv_data_display.c 
    )


message("adding cxx-utilities library")
blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 HEADERS_OUTPUT_SUBDIR cxx-utilities
                 DEPENDS_ON       ${cxx_dependencies}
                 )

target_include_directories( cxx-utilities PUBLIC ${CMAKE_BINARY_DIR}/include )

