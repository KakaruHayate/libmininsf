cmake_minimum_required(VERSION 3.16)

project(libmininsf
    VERSION 0.1.0
    DESCRIPTION "Small C implementation of DiffSinger MiniNSF fastsinegen"
    LANGUAGES C CXX
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(MININSF_BUILD_TESTS "Build tests" ON)
option(MININSF_BUILD_EXAMPLES "Build native examples" ON)
option(MININSF_USE_OPENMP "Use OpenMP when available" ON)
option(MININSF_NATIVE "Enable native CPU code generation on supported compilers" OFF)

add_library(mininsf src/mininsf.c)
add_library(mininsf::mininsf ALIAS mininsf)

target_include_directories(mininsf
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(mininsf PUBLIC c_std_99)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(mininsf PRIVATE MININSF_BUILD_SHARED)
endif()

if(MSVC)
    target_compile_options(mininsf PRIVATE /W4)
else()
    target_compile_options(mininsf PRIVATE -Wall -Wextra)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(mininsf PRIVATE -O3)
    endif()
    if(MININSF_NATIVE)
        target_compile_options(mininsf PRIVATE -march=native)
    endif()
endif()

if(MININSF_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(mininsf PRIVATE OpenMP::OpenMP_C)
        target_compile_definitions(mininsf PRIVATE MININSF_HAS_OPENMP=1)
    endif()
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(mininsf PUBLIC m)
endif()

install(TARGETS mininsf
    EXPORT mininsfTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT mininsfTargets
    FILE mininsfTargets.cmake
    NAMESPACE mininsf::
    DESTINATION lib/cmake/mininsf
)

if(MININSF_BUILD_TESTS)
    enable_testing()
    add_executable(test_mininsf tests/test_mininsf.c)
    target_link_libraries(test_mininsf PRIVATE mininsf::mininsf)
    add_test(NAME test_mininsf COMMAND test_mininsf)
endif()

if(MININSF_BUILD_EXAMPLES)
    add_executable(write_source_c examples/write_source.c)
    target_link_libraries(write_source_c PRIVATE mininsf::mininsf)

    add_executable(write_source_cpp examples/cpp/write_source.cpp)
    target_link_libraries(write_source_cpp PRIVATE mininsf::mininsf)
    target_compile_features(write_source_cpp PRIVATE cxx_std_11)
endif()
