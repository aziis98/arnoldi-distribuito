cmake_minimum_required(VERSION 3.5.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

# Remove the lines below if you do not wish to have PETSc determine the compilers
SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_Fortran_COMPILER gfortran)
enable_language(Fortran)

# Tells CMake to build the application ex1 from the source file ex1.c
# This must appear AFTER the compilers are set
project(main)
add_executable(main main.c)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(main MPI::MPI_C MPI::MPI_CXX MPI::MPI_Fortran PkgConfig::PETSC)
