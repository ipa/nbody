#include <cuda.h>
#include <stdio.h>
#include <device_types.h>
#include <vector_types.h>
#include <math.h>
#include <pthread.h>


#define EPS2 0.67f
#define BSIZE 256
#define DAMPING 0.999f
#define DAMPINT_BOUNDARY 0.9f

#define CHECK_BOUNDARY 0

float3 calc_accel_body_cpu(float4 bi, float4 bj, float3 ai) 
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

float3 calc_accel_cpu(float4* shPosition, float4 myPosition, float3 accel) 
{
	int i;
	for (i = 0; i < BSIZE; i++) {
		accel = calc_accel_body_cpu(myPosition, shPosition[i], accel);
	}
	return accel;
}

void galaxyKernel_cpu(float4 * pdata, unsigned int bodies, float step) 
{
	// index of my body
	unsigned int x;
	for(x = 0; x < bodies; x++){
		unsigned int pLoc = x;
		unsigned int vLoc = bodies + pLoc;

		float4 myPosition = pdata[pLoc];
		float4 myVelocity = pdata[vLoc];

		float3 acc = { 0.0f, 0.0f, 0.0f };
		unsigned int loop = bodies;
		unsigned int i;

		for (i = 0; i < loop; i++) {
			acc = calc_accel_cpu(pdata, myPosition, acc);
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
}

float4* tmp_pdata;
int tmp_N;
float tmp_step;

void galaxyKernel_cpu_t(void *ptr)
{
	float4* pdata = tmp_pdata;
	int N = tmp_N;
	float step = tmp_step;
	galaxyKernel_cpu(pdata, N, step);
}

extern void cComputeGalaxy(float4 * pdata, int N, float step) 
{
	tmp_pdata = pdata;
	tmp_step = step;
	tmp_N = N;

	pthread_t thread1;
	int  iret1;
	
	iret1 = pthread_create( &thread1, NULL, galaxyKernel_cpu_t, (void*)"hello");
	pthread_join( thread1, NULL);
	
	//galaxyKernel_cpu(pdata, N, step);
	cudaThreadSynchronize();
}
