PETSC_DIR = ./deps/petsc
PETSC_ARCH = arch-linux-c-debug

petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

PACKAGES := $(petsc.pc)

CC := $(shell pkg-config --variable=ccompiler $(PACKAGES))
CXX := $(shell pkg-config --variable=cxxcompiler $(PACKAGES))
FC := $(shell pkg-config --variable=fcompiler $(PACKAGES))
CFLAGS_OTHER := $(shell pkg-config --cflags-only-other $(PACKAGES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
CXXFLAGS := $(shell pkg-config --variable=cxxflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
FFLAGS := $(shell pkg-config --variable=fflags_extra $(PACKAGES))
CPPFLAGS := $(shell pkg-config --cflags-only-I $(PACKAGES))
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, $(shell pkg-config --libs-only-L $(PACKAGES)))
LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm
CUDAC := $(shell pkg-config --variable=cudacompiler $(PACKAGES))
CUDAC_FLAGS := $(shell pkg-config --variable=cudaflags_extra $(PACKAGES))
CUDA_LIB := $(shell pkg-config --variable=cudalib $(PACKAGES))
CUDA_INCLUDE := $(shell pkg-config --variable=cudainclude $(PACKAGES))

all: setup deps ./bin/main

setup:
	mkdir -p bin
	mkdir -p deps

deps: deps/petsc deps/psblas

deps/%:
	cd $@ && ./configure
	cd $@ && make	

print:
	@echo ====== PETSc ======
	@echo PETSC_DIR=$(PETSC_DIR)
	@echo PETSC_ARCH=$(PETSC_ARCH)
	@echo ====== Compilers ======
	@echo CC=$(CC)
	@echo CXX=$(CXX)
	@echo FC=$(FC)
	@echo COMPILE.cc=$(COMPILE.cc)
	@echo LINK.cc=$(LINK.cc)
	@echo ====== Flags ======
	@echo CFLAGS=$(CFLAGS)
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo FFLAGS=$(FFLAGS)
	@echo CPPFLAGS=$(CPPFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
	@echo ====== Cuda ======
	@echo CUDAC=$(CUDAC)
	@echo CUDAC_FLAGS=$(CUDAC_FLAGS)
	@echo CUDA_LIB=$(CUDA_LIB)
	@echo CUDA_INCLUDE=$(CUDA_INCLUDE)

bin/%: %.c
	$(LINK.cc) -o $@ $^ $(LDLIBS)
bin/%.o: %.f90
	$(FC) $(OUTPUT_OPTION) $<
bin/%.o: %.cxx
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
bin/%.o: %.c
	$(CC) $(OUTPUT_OPTION) $<
bin/%.o : %.cu
	$(CUDAC) -c $(CPPFLAGS) $(CUDAC_FLAGS) $(CUDA_INCLUDE) -o $@ $<
