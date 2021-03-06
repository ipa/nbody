#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <device_types.h>
#include <vector_types.h>
#include <stdbool.h>
#include <string.h>

#include "display.h"

#define MAXSTR 256

// simulation parameter
float scaleFactor = 1.0f;
float velFactor = 8.0f; 
float massFactor = 0.0001f; 
float gStep = 0.025f; 

bool gpu = true;

int numBodies = 1024;
bool randData = false;

// simulation data storage
float* gPos = 0;
float* gVel = 0;
float* d_particleData = 0; // device side particle data storage
float* h_particleData = 0; // host side particle data storage

// cuda related...
int numBlocks = 1;
int numThreadsPerBlock = 256;

// functions
void loadData(char* filename, int bodies);
void init(int bodies);
void runCuda(void);
void loadDataRand(int bodies);

void check() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
		exit(-1);
	}
}

int count = 0;
int main( int argc, char *argv[] ) {
	initDisplay();
	// get number of SMs on this GPU
	int devID;
	/*cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);
	check();

	printf("using device %s \n", props.name);
	printf("number of MPs %d \n", props.multiProcessorCount);
	printf("max threads per block %d \n", props.maxThreadsPerBlock);
	printf("max concurrent kernels %d \n", props.concurrentKernels);
	printf("number of bodies %d \n", numBodies);
*/

	if(argc == 2)
	{
		if(strcmp(argv[1], "--cpu") == 0)
		{
			gpu = false;
		}
		else if(strcmp(argv[1], "--gpu") == 0)
		{
			gpu = true;
		}
		else
		{
			printf("wrong argument %s \nallowed: NULL, --cpu, --gpu \n", argv[1]);
			exit(EXIT_FAILURE);
		}
	}

	if(randData){
		loadDataRand(numBodies);
	}else{
		loadData("data/data.tab", numBodies);
	}
	init(numBodies);

	printf("stop with CTRL+C\n");
	
	do {
		runCuda();
		showGalaxy(h_particleData, numBodies, false);
		//count++;
	} while (count < 64);

	printf("finished...\n");

	if(gpu)
	{
		cudaFree(d_particleData);
	}
	closeWindow();
	printf("close...\n");
	return EXIT_SUCCESS;
}

void loadDataRand(int bodies) {
	gPos = (float*) malloc(sizeof(float) * bodies * 4);
	gVel = (float*) malloc(sizeof(float) * bodies * 4);
	int i;
	for (i = 0; i < bodies; i++) {
		int idx = i * 4;
		// random init
		gPos[idx + 0] = 500 / 2;
		gPos[idx + 1] = 500 / 2;
		gPos[idx + 2] = 500 / 2;
		gPos[idx + 3] = 100.0f;
		float vx = random();
		float vy = random();
		float dx = (random() - random()) < 0 ? -1 : +1;
		float dy = (random() - random()) < 0 ? -1 : +1;
		vx = vx * 0.0000000000011f * dx;
		vy = vy * 0.0000000000011f * dy;

		gVel[idx + 0] = vx;
		gVel[idx + 1] = vy;
		gVel[idx + 2] = 0;
		gVel[idx + 3] = 1.0f;
	}
}

void loadData(char* filename, int bodies) {
	FILE *fin;
	if ((fin = fopen(filename, "r"))) {

		char buf[MAXSTR];
		float v[7];
		register int idx = 0;

		// allocate memory
		gPos = (float*) malloc(sizeof(float) * bodies * 4);
		gVel = (float*) malloc(sizeof(float) * bodies * 4);

		int i;
		for (i = 0; i < bodies; i++) {

			// depend on input size...
			fgets(buf, MAXSTR, fin);
			sscanf(buf, "%f %f %f %f %f %f %f", v + 0, v + 1, v + 2, v + 3,
					v + 4, v + 5, v + 6);

			// update index
			idx = i * 4;

			// position
			gPos[idx + 0] = abs(v[1]) * scaleFactor;
			gPos[idx + 1] = abs(v[2]) * scaleFactor;
			gPos[idx + 2] = abs(v[3]) * scaleFactor;
			// mass
			gPos[idx + 3] = abs(v[0]) * massFactor;
			// velocity
			gVel[idx + 0] = abs(v[4]) * velFactor;
			gVel[idx + 1] = abs(v[5]) * velFactor;
			gVel[idx + 2] = abs(v[6]) * velFactor;
			gVel[idx + 3] = 1.0f;
		}

	} else {
		printf("cannot find file...: %s\n", filename);
		exit(EXIT_FAILURE);
	}
}

void init(int bodies) {
	// blocks per grid
	numBlocks = bodies / numThreadsPerBlock;

	// host particle data (position, velocity
	h_particleData = (float *) malloc(8 * bodies * sizeof(float));

	// device particle data
	if(gpu)
	{
		cudaMalloc((void**) &d_particleData, 8 * bodies * sizeof(float));
		check();
	}
	// load inital data set
	int idx = 0;
	int vidx = 0;
	int i;
	for (i = 0; i < bodies; i++) {
		// float array index
		idx = i * 4;
		vidx = bodies * 4 + idx;

		// set value from global data storage
		float randX = (random() - random()) * 0.00000002f;
		float randY = (random() - random()) *  0.00000002f;
		h_particleData[idx + 0] = gPos[idx + 0]; // x
		h_particleData[idx + 1] = gPos[idx + 1]; // y
		h_particleData[idx + 2] = gPos[idx + 2]; // z
		h_particleData[idx + 3] = gPos[idx + 3]; // mass
		h_particleData[vidx + 0] = gVel[idx + 0] + randX; // vx
		h_particleData[vidx + 1] = gVel[idx + 1] + randY; // vy
		h_particleData[vidx + 2] = gVel[idx + 2]; // vz
		h_particleData[vidx + 3] = gVel[idx + 3]; // padding
	}

	if(gpu)
	{
		// copy initial value to GPU memory
		cudaMemcpy(d_particleData, h_particleData, 8 * bodies * sizeof(float),
				cudaMemcpyHostToDevice);
		check();
	}

}

extern void cudaComputeGalaxy(float4 * pdata, int N, float step);
extern void cComputeGalaxy(float4 * pdata, int N, float step);

void runCuda(void) {
	if(gpu)
	{
		cudaComputeGalaxy((float4*) d_particleData, numBodies, gStep);
		check();

		cudaMemcpy(h_particleData, d_particleData, 4 * numBodies * sizeof(float),
		cudaMemcpyDeviceToHost);
		check();
	}
	else
	{
		cComputeGalaxy((float4*) h_particleData, numBodies, gStep);
	}
}
