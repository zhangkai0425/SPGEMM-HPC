CUDAC = nvcc

# CUDAOPT = 
CUDAOPT = -O3 -use_fast_math

CUDAFLAGS = $(CUDAOPT)

CUDAPATH = /usr/local/cuda/lib64
LDFLAGS = 
LDLIBS = $(LDFLAGS) -L$(CUDAPATH) -L$(CUDAPATH)/stubs -lcuda -lcudart

targets =  benchmark-cusparse benchmark-optimized
commonobj = utils.o benchmark.o
objects = $(commonobj) spgemm-cusparse.o spgemm-optimized.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark.o : benchmark.cu common.h utils.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
utils.o : utils.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@

spgemm-cusparse.o : spgemm-cusparse.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-cusparse : $(commonobj) spgemm-cusparse.o
	$(CUDAC) -o $@ $^ $(LDLIBS) -lcusparse

spgemm-optimized.o : spgemm-optimized-v1.cu common.h
	$(CUDAC) -c $(CUDAFLAGS) $< -o $@
benchmark-optimized : $(commonobj) spgemm-optimized.o
	$(CUDAC) -o $@ $^ $(LDLIBS) -lcusparse

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)
