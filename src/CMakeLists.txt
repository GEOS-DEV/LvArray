add_subdirectory(docs)

    
        set(lvarray_headers
    templateHelpers.hpp
    Array.hpp
    ArraySlice.hpp
    streamIO.hpp
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
    NewChaiBuffer.hpp
    MallocBuffer.hpp
    tensorOps.hpp
    sliceHelpers.hpp
   )

set(lvarray_sources
    SetFPE.cpp
    SetSignalHandling.cpp
    stackTrace.cpp
    StringUtilities.cpp
    totalview/tv_data_display.c 
    )


message("adding lvarray library")
blt_add_library( NAME             lvarray
                 SOURCES          ${lvarray_sources}
                 HEADERS          ${lvarray_headers}
                 DEPENDS_ON       ${lvarray_dependencies}
                 )

target_include_directories( lvarray PUBLIC ${CMAKE_BINARY_DIR}/include )

lvarray_add_code_checks( PREFIX lvarray
                         EXCLUDES "blt/*" )

