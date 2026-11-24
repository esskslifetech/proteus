cmake_minimum_required(VERSION 3.20)
project(proteus-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /permissive- /std:c++20)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -std=c++20)
endif()

# Find dependencies
find_package(Eigen3 REQUIRED)
# SUNDIALS is optional: without it the ODE layer uses an RK4 fallback.
find_package(Sundials QUIET COMPONENTS cvode ida kinsol nvector_serial sunmatrix_dense sunlinsol_dense sunlinsol_spgmr)
set(proteus_USE_SUNDIALS ${Sundials_FOUND})
# HiGHS is optional - if not found, we'll use Eigen for linear programming
find_package(HiGHS QUIET)
set(proteus_USE_HIGHS ${HiGHS_FOUND})

# pybind11
find_package(pybind11 REQUIRED)

# Shared library for kernels
add_library(proteus_kernels SHARED
    kernels/src/simd_stats.cpp
    kernels/src/energy_functions.cpp
)

target_include_directories(proteus_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels/include>
    $<INSTALL_INTERFACE:include/proteus/kernels>
)
target_link_libraries(proteus_kernels PUBLIC Eigen3::Eigen)

# On Windows/MSVC, a SHARED library only produces an import library for
# exported symbols. The kernels have no explicit __declspec annotations, so
# export everything; the pybind11 module links against these import libraries.
set_target_properties(proteus_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Solvers library
add_library(proteus_solvers SHARED
    solvers/src/ode_solver.cpp
    solvers/src/fba_solver.cpp
)

target_include_directories(proteus_solvers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/solvers/include>
    $<INSTALL_INTERFACE:include/proteus/solvers>
)
target_link_libraries(proteus_solvers PUBLIC Eigen3::Eigen)

set_target_properties(proteus_solvers PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(proteus_USE_SUNDIALS)
    target_link_libraries(proteus_solvers PUBLIC
        SUNDIALS::cvode
        SUNDIALS::ida
        SUNDIALS::kinsol
        SUNDIALS::nvecserial
        SUNDIALS::sunmatrixdense
        SUNDIALS::sunlinsoldense
        SUNDIALS::sunlinsolspgmr
    )
endif()

# HiGHS support (optional)
if(proteus_USE_HIGHS)
    target_include_directories(proteus_solvers PUBLIC ${HiGHS_INCLUDE_DIRS})
    target_link_libraries(proteus_solvers PUBLIC highs::highs)
endif()

target_compile_definitions(proteus_solvers PUBLIC
    proteus_USE_HIGHS=$<BOOL:${proteus_USE_HIGHS}>
    proteus_USE_SUNDIALS=$<BOOL:${proteus_USE_SUNDIALS}>
)

# Python bindings
pybind11_add_module(proteus_cpp bindings/src/bindings.cpp)
target_link_libraries(proteus_cpp PRIVATE proteus_kernels proteus_solvers)

# Install targets
install(TARGETS proteus_kernels proteus_solvers
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY kernels/include/ DESTINATION include/proteus/kernels FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY solvers/include/ DESTINATION include/proteus/solvers FILES_MATCHING PATTERN "*.hpp")

# Tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    enable_testing()
    add_subdirectory(tests)
endif()