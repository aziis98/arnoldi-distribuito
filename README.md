# Arnoldi Iteration

## Setup

### Spack

```bash shell
$ spack env activate ./spack-env
$ spack install
$ spack load openblas openmpi petsc
```

### CMake

To build main from main.c

```bash shell
$ rm -rf build
$ mkdir build
$ cd build
$ cmake ..
$ make
$ mpirun -n 2 ./main
```
