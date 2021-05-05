#include "LBM.h"
#include "dados.h"
#include "boundary.h"

using namespace myGlobals;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_bounce_back(double*);
__global__ void gpu_outlet(double, double*);

// Boundary Conditions
__device__ void device_bounce_back(unsigned int x, unsigned int y, double *f){
	
	if(y == 0){
		f[gpu_fieldn_index(x, y, 2)] = f[gpu_fieldn_index(x, y, 4)];
		f[gpu_fieldn_index(x, y, 5)] = f[gpu_fieldn_index(x, y, 7)];
		f[gpu_fieldn_index(x, y, 6)] = f[gpu_fieldn_index(x, y, 8)];

		f[gpu_fieldn_index(x, y, 9)] = f[gpu_fieldn_index(x+1, y+1, 11)];
		f[gpu_fieldn_index(x, y, 10)] = f[gpu_fieldn_index(x-1, y+1, 12)];

		f[gpu_fieldn_index(x, y, 14)] = f[gpu_fieldn_index(x, y+2, 16)];
	}
	else if(y == 1){
		f[gpu_fieldn_index(x, y, 9)] = f[gpu_fieldn_index(x-1, y-1, 11)];
		f[gpu_fieldn_index(x, y, 10)] = f[gpu_fieldn_index(x+1, y-1, 12)];

		f[gpu_fieldn_index(x, y, 14)] = f[gpu_fieldn_index(x, y, 16)];
	}
	else if(y == 2){
		f[gpu_fieldn_index(x, y, 14)] = f[gpu_fieldn_index(x, y-2, 16)];
	}

	if(y == Ny_d-1){
		f[gpu_fieldn_index(x, y, 4)] = f[gpu_fieldn_index(x, y, 2)];
		f[gpu_fieldn_index(x, y, 7)] = f[gpu_fieldn_index(x, y, 5)];
		f[gpu_fieldn_index(x, y, 8)] = f[gpu_fieldn_index(x, y, 6)];

		f[gpu_fieldn_index(x, y, 11)] = f[gpu_fieldn_index(x-1, y-1, 9)];
		f[gpu_fieldn_index(x, y, 12)] = f[gpu_fieldn_index(x+1, y-1, 10)];

		f[gpu_fieldn_index(x, y, 16)] = f[gpu_fieldn_index(x, y-2, 14)];
	}
	else if(y == Ny_d-2){
		f[gpu_fieldn_index(x, y, 11)] = f[gpu_fieldn_index(x+1, y+1, 9)];
		f[gpu_fieldn_index(x, y, 12)] = f[gpu_fieldn_index(x-1, y+1, 10)];

		f[gpu_fieldn_index(x, y, 16)] = f[gpu_fieldn_index(x, y, 14)];
	}
	else if(y == Ny_d-3){
		f[gpu_fieldn_index(x, y, 16)] = f[gpu_fieldn_index(x, y+2, 14)];
	}
}

__host__ void bounce_back(double *f){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_bounce_back<<< grid, block >>>(f);
	getLastCudaError("gpu_bounce_back kernel error");
}

__global__ void gpu_bounce_back(double *f){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_walls = walls_d[gpu_scalar_index(x, y)];
	if(node_walls){
		device_bounce_back(x, y, f);
	}
}
/*
__host__ void inlet_BC(double u_def, double *f, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_inlet<<< grid, block >>>(u_def, f, r, u, v);
	getLastCudaError("gpu_inlet kernel error");
}

__global__ void gpu_inlet(double udef, double *f, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x == 0){

		unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};

		double rhoI = 0.0, rhoaxy = 0.0;
		for(int n = 0; n < q; ++n){
			unsigned int ni = I[n];
			rhoI += f[gpu_fieldn_index(x, y, ni)];
			rhoaxy += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
		}

		double udef2 = u_def*u_def;
		double udef3 = u_def*u_def*u_def;

		double rho = (129600*rhoI)/((5*sqrt(193)+5525)*udef3 + (-31380-1740*sqrt(193))*udef2 + (-54144-720*sqrt(193))*udef + (808*sqrt(193)+94712));
		double tauxy = (-270*rhoaxy)/((4*sqrt(193)+190)*udef - 135);
	}
}
*/
__host__ void outlet_BC(double rho, double *f){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_outlet<<< grid, block >>>(rho, f);
	getLastCudaError("gpu_outlet kernel error");
}

__global__ void gpu_outlet(double rho, double *f){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_outlet = outlet_d[gpu_scalar_index(x, y)];
	if(node_outlet){

		double sumRho = 0.0;
		for(int n = 0; n < q; ++n){
			sumRho += f[gpu_fieldn_index(x-3, y, n)];
		}


		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = (rho/sumRho)*f[gpu_fieldn_index(x-3, y, n)];
		}
	}
}