#include <cuda.h>
#include <stdio.h>

#define EPS2 0.67f
#define BSIZE 256
#define DAMPING 0.999f
#define DAMPINT_BOUNDARY 0.9f

#define CHECK_BOUNDARY 0

__device__  __shared__ float4 shPosition[256];

__device__ float3 calc_accel_body(float4 bi, float4 bj, float3 ai) 
{
	float3 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = 1.0f / sqrtf(distSixth);

	float s = bj.w * invDistCube;

	ai.x += r.x * s;
	ai.y += r.y * s;
	ai.z += r.z * s;
	return ai;
}

__device__ float3 calc_accel(float4 myPosition, float3 accel) 
{
	int i;
	extern __shared__ float4 shPosition[];

	for (i = 0; i < BSIZE; i++) {
		accel = calc_accel_body(myPosition, shPosition[i], accel);
	}
	return accel;
}

__global__ void galaxyKernel(float4 * pdata, unsigned int bodies, float step) 
{
	// shared memory
	extern __shared__ float4 shPosition[];

	// index of my body
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int pLoc = y * gridDim.y * blockDim.y + x;
	unsigned int vLoc = bodies + pLoc;

	float4 myPosition = pdata[pLoc];
	float4 myVelocity = pdata[vLoc];

	float3 acc = { 0.0f, 0.0f, 0.0f };
	unsigned int loop = gridDim.x * gridDim.y;

	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	for (int i = 0; i < loop; i++) {
		shPosition[idx] = pdata[idx + BSIZE * i];
		__syncthreads();

		acc = calc_accel(myPosition, acc);
		__syncthreads();
	}

	// update velocity with above acc
	myVelocity.x += acc.x * step; // * 2.0f;
	myVelocity.y += acc.y * step; // * 2.0f;
	myVelocity.z += acc.z * step; // * 2.0f;

	// damping
	myVelocity.x *= DAMPING;
	myVelocity.y *= DAMPING;
	myVelocity.z *= DAMPING;

	// update position
	myPosition.x += myVelocity.x * step;
	myPosition.y += myVelocity.y * step;
	myPosition.z += myVelocity.z * step;

	__syncthreads();
#if CHECK_BOUNDARY
	if(myPosition.x > 650.0f || myPosition.x < -150.0f){
		myVelocity.x *= -1.0f * DAMPINT_BOUNDARY;
	}
	if(myPosition.y > 650.0f || myPosition.y < -150.0f){
		myVelocity.y *= -1.0f * DAMPINT_BOUNDARY;
	}
#endif
	// update device memory
	pdata[pLoc] = myPosition;
	pdata[vLoc] = myVelocity;
}

extern "C" void cudaComputeGalaxy(float4 * pdata, int N, float step) 
{
	dim3 block(16, 16, 1);
	int dim = sqrt(N / BSIZE);
	dim3 grid(dim ,dim ,1);

	grid.y = grid.y == 0 ? 1 : grid.y;
	grid.x = grid.x == 0 ? 1 : grid.x;

	galaxyKernel<<<grid, block>>>(pdata, N, step);

	cudaThreadSynchronize();
}
