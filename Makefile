CC=gcc
MPICC=mpicc
CILKCC=/usr/local/OpenCilk-9.0.1-Linux/bin/clang
CFLAGS=-O3

default: all

V0:
	$(CC) $(CFLAGS) -o V0 V0.c -lopenblas -lm -fopenmp

V1:
	$(MPICC) $(CFLAGS) -o V1 V1.c -lopenblas -lm -fopenmp

V2:
	$(MPICC) $(CFLAGS) -o V2 V2.c  -lm -fopenmp

.PHONY: clean

all: V0 V1 V2


	
p=0
testV1:
	@printf "\n** Testing V1 **\n\n"
	mpiexec -np $(p) ./V1
testV2:
	@printf "\n** Testing V2 **\n\n"
	mpiexec -np $(p) ./V2

clean:
	rm -f V0 V1 V2


