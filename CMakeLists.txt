cmake_minimum_required(VERSION 3.16)

# Enable compile_commands.json generation for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)

project(main)

find_package(PkgConfig REQUIRED)

pkg_search_module(OpenBLAS REQUIRED IMPORTED_TARGET openblas)
pkg_search_module(PETSc REQUIRED IMPORTED_TARGET petsc)
pkg_search_module(hdf5 REQUIRED IMPORTED_TARGET hdf5)
pkg_search_module(OpenMPI REQUIRED IMPORTED_TARGET ompi)

# add_executable(main main.c)
# target_include_directories(main PUBLIC ${PETSc_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIRS} ${OpenMPI_INCLUDE_DIRS})
# target_link_libraries(main PUBLIC m PkgConfig::PETSc PkgConfig::OpenBLAS PkgConfig::OpenMPI)

add_executable(arnoldi arnoldi.c)
target_include_directories(arnoldi PUBLIC 
    ${PETSc_INCLUDE_DIRS}
    ${hdf5_INCLUDE_DIRS}
    ${OpenBLAS_INCLUDE_DIRS}
    ${OpenMPI_INCLUDE_DIRS}
)
target_link_libraries(arnoldi PUBLIC 
    m
    PkgConfig::PETSc
    PkgConfig::hdf5
    PkgConfig::OpenBLAS
    PkgConfig::OpenMPI
)

# add_executable(ex10 ex10.c)
# target_include_directories(ex10 PUBLIC ${PETSc_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIRS} ${OpenMPI_INCLUDE_DIRS})
# target_link_libraries(ex10 PUBLIC m PkgConfig::PETSc PkgConfig::OpenBLAS PkgConfig::OpenMPI)
