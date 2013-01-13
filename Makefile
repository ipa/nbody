
all: nbody

rebuild: clean nbody

# define paths
GCC_LINKER = -L/usr/local/cuda/lib -L/opt/X11/lib
GCC_LIBS = -lX11 -lcudart
CUDA_INCLUDE = -I/usr/local/cuda/extras/CUPTI/include -I/usr/local/cuda/include 
X11_INCLUDE = -I/opt/X11/include
ARCH = -m64
O = -O3

nbody: nbody.o display.o main.o cpunbody.o
	gcc -o nbody --link $(GCC_LINKER) $(GCC_LIBS) $(ARCH) $(O) nbody.o cpunbody.o display.o main.o 

cpunbody.o: cpunbody.c
	gcc -o cpunbody.o $(CUDA_INCLUDE) $(ARCH) $(O) -c cpunbody.c

main.o: main.c
	gcc -o main.o $(CUDA_INCLUDE) $(ARCH) $(O) -c main.c

display.o: display.c display.h 
	gcc -o display.o $(X11_INCLUDE) $(ARCH) $(O) -c display.c 

nbody.o: nbody.cu
	nvcc -o nbody.o $(ARCH) $(O) -c nbody.cu

run: nbody
	./nbody

clean:
	rm -f nbody.o display.o main.o nbody