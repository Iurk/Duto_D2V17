#include "LBM.h"
#include "boundary.h"

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

// Boundary Conditions
__device__ void gpu_bounce_back(unsigned int x, unsigned int y, double *f){
	
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