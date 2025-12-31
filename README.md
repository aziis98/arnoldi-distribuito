# Arnoldi Iteration for Eigenvalues

Project for the course "Scientific Computing" and the special ["Parallel Computing"](https://numpi.dm.unipi.it/teaching/calcolo-parallelo-dallinfrastruttura-alla-matematica-2022-23/) of our department. I run this project on the following beowulf cluster we build during the parallel computing course.

<div align="center">
  <img src="https://github.com/user-attachments/assets/292ee2ef-c47e-4b8a-94e7-7eee7599c4c0" alt="Description 1" width="45%">
  <img src="https://github.com/user-attachments/assets/e0195261-6222-43d4-867c-2adfeaea25fb" alt="Description 2" width="45%">
  <p align="center">
    <em>The cluster we build for the parallel computing project with before and after the case-change. This holds about ~25 [Rock 4C+ boards](https://radxa.com/products/rock4/4cp). Currently hosted in PHC</em>
  </p>
</div>

---

Download the
[report](https://raw.githubusercontent.com/aziis98/arnoldi-distribuito/refs/heads/next/typst-report/main.pdf)

## Setup

### Spack

> I recently wrote
> [an article about this](https://aziis98.com/blog/using-spack/)

This assumes that you have spack installed and sourced. If not, you can install
it using the following commands

```bash shell
$ git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git

# For bash/zsh/sh
$ . spack/share/spack/setup-env.sh
# For tcsh/csh
$ source spack/share/spack/setup-env.csh
# For fish
$ . spack/share/spack/setup-env.fish

# Then to discover available compilers
$ spack compiler list
$ spack compiler find
```

Now in the root directory of this repository, you can load the spack environment
that contains the packages needed to build the project.

```bash shell
# 1. Activate the local environment
$ spack env activate ./spack-env

# 2. Compile and install the packages listed in the environment
$ spack install

# 3. (optional, it looks like cmake can still find the packages just by loading the environment using pkg-config)
$ spack load lapack openmpi petsc
```

### CMake

To build `main` from `main.c`

```bash shell
$ rm -rf build
$ mkdir build
$ cd build
$ cmake ..
$ make
$ mpirun -n 2 ./main
```
