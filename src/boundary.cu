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

__global__ void gpu_inlet(double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, unsigned int);
__global__ void gpu_bounce_back(double*);
__global__ void gpu_outlet(double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, unsigned int);
__global__ void gpu_wall_velocity(double*, double*, double*, double*, double*, double*, double*, double*, double*);

// Moments
__device__ void known_moments(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *moments){

	double rhoI = 0.0, rhoaxy = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni= I[n];

		rhoI += f[gpu_fieldn_index(x, y, ni)];
		rhoaxy += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
	}

	moments[0] = rhoI;
	moments[1] = rhoaxy;
}

__device__ void att_moments(unsigned int x, unsigned int y, double *f, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;

	double rho = 0, ux_i = 0, uy_i = 0, Pxx = 0, Pxy = 0, Pyy = 0;
	for(int n = 0; n < q; ++n){
		rho += f[gpu_fieldn_index(x, y, n)];
		ux_i += f[gpu_fieldn_index(x, y, n)]*ex_d[n];
		uy_i += f[gpu_fieldn_index(x, y, n)]*ey_d[n];
		Pxx += f[gpu_fieldn_index(x, y, n)]*(ex_d[n]*ex_d[n] - cs2);
		Pxy += f[gpu_fieldn_index(x, y, n)]*ex_d[n]*ey_d[n];
		Pyy += f[gpu_fieldn_index(x, y, n)]*(ey_d[n]*ey_d[n] - cs2);
	}

	double ux = ux_i/rho;
	double uy = uy_i/rho;
	
	double tauxx = Pxx - rho*ux*ux;
	double tauxy = Pxy - rho*ux*uy;
	double tauyy = Pyy - rho*uy*uy;

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
	txx[gpu_scalar_index(x, y)] = tauxx;
	txy[gpu_scalar_index(x, y)] = tauxy;
	tyy[gpu_scalar_index(x, y)] = tauyy;
}

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

__device__ void device_inlet_VP(unsigned int x, unsigned int y, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double moments[2];
	double uy_in = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	ux_in = poiseulle_eval(x, y);

	if(x == 0){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};

		known_moments(x, y, NI, I, f, moments);

		double rhoI = moments[0];
		double rhoaxy = moments[1];

		double ux_in2 = ux_in*ux_in;
		double ux_in3 = ux_in*ux_in*ux_in;

		rho = (129600*rhoI)/((5*sqrt(193.0)+5525)*ux_in3 + (-31380-1740*sqrt(193.0))*ux_in2 + (-54144-720.0*sqrt(193.0))*ux_in + (808*sqrt(193.0)+94712));
		tauxy = (-270*rhoaxy)/((4*sqrt(193.0)+190)*ux_in - 135);

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux_in;
		v[gpu_scalar_index(x, y)] = uy_in;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		gpu_recursive(x, y, rho, ux_in, uy_in, tauxx, tauxy, tauyy, frec);

		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
		}
	}

	else if(x == 1){
		f[gpu_fieldn_index(x, y, 9)] = f[gpu_fieldn_index(x, y, 11)] - feq[gpu_fieldn_index(x, y, 11)] + feq[gpu_fieldn_index(x, y, 9)];
		f[gpu_fieldn_index(x, y, 12)] = f[gpu_fieldn_index(x, y, 10)] - feq[gpu_fieldn_index(x, y, 10)] + feq[gpu_fieldn_index(x, y, 12)];

		if(y > 0 && y < Ny_d-2){
			f[gpu_fieldn_index(x, y, 9)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y-1, 9)] + f[gpu_fieldn_index(x+1, y+1, 9)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y+2, 9)];
		}

		if(y > 1 && y < Ny_d-1){
			f[gpu_fieldn_index(x, y, 12)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y+1, 12)] + f[gpu_fieldn_index(x+1, y-1, 12)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y-2, 12)];
		}
		
		f[gpu_fieldn_index(x, y, 13)] = (1.0/2.0)*f[gpu_fieldn_index(x-1, y, 13)] + f[gpu_fieldn_index(x+2, y, 13)] - (1.0/2.0)*f[gpu_fieldn_index(x+3, y, 13)];
	}

	else if(x == 2){
		f[gpu_fieldn_index(x, y, 13)] = (1.0/6.0)*f[gpu_fieldn_index(x-2, y, 13)] + (4.0/3.0)*f[gpu_fieldn_index(x+1, y, 13)] - (1.0/2.0)*f[gpu_fieldn_index(x+2, y, 13)];
	}
}

__device__ void device_outlet_VP(unsigned int x, unsigned int y, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double*txx, double *txy, double *tyy){

	double moments[2];
	double uy_in = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	ux_in = poiseulle_eval(x, y);

	if(x == Nx_d-1){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 2, 4, 5, 8, 9, 12, 13, 14, 16};

		known_moments(x, y, NI, I, f, moments);

		double rhoI = moments[0];
		double rhoaxy = moments[1];

		double ux_in2 = ux_in*ux_in;
		double ux_in3 = ux_in*ux_in*ux_in;

		rho = (-129600*rhoI)/(5594.46221994725*ux_in3 + 55552.8525416427*ux_in2 - 64146.5596724039*ux_in - 105937.094743475);
		tauxy = (270*rhoaxy)/(245.569775957799*ux_in + 135);

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux_in;
		v[gpu_scalar_index(x, y)] = uy_in;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		gpu_recursive(x, y, rho, ux_in, uy_in, tauxx, tauxy, tauyy, frec);

		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
		}
	}

	else if(x == Nx_d-2){
		f[gpu_fieldn_index(x, y, 10)] = f[gpu_fieldn_index(x, y, 12)] - feq[gpu_fieldn_index(x, y, 12)] + feq[gpu_fieldn_index(x, y, 10)];
		f[gpu_fieldn_index(x, y, 11)] = f[gpu_fieldn_index(x, y, 9)] - feq[gpu_fieldn_index(x, y, 9)] + feq[gpu_fieldn_index(x, y, 11)];

		if(y > 0 && y < Ny_d-2){
			f[gpu_fieldn_index(x, y, 10)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y-1, 10)] + f[gpu_fieldn_index(x-1, y+1, 10)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y+2, 10)];
		}

		if(y > 1 && y < Ny_d-1){
			f[gpu_fieldn_index(x, y, 11)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y+1, 11)] + f[gpu_fieldn_index(x-1, y-1, 11)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y-2, 11)];
		}
		
		f[gpu_fieldn_index(x, y, 15)] = (1.0/2.0)*f[gpu_fieldn_index(x+1, y, 15)] + f[gpu_fieldn_index(x-2, y, 15)] - (1.0/2.0)*f[gpu_fieldn_index(x-3, y, 15)];
	}

	else if(x == Nx_d-3){
		f[gpu_fieldn_index(x, y, 15)] = (1.0/6.0)*f[gpu_fieldn_index(x+2, y, 15)] + (4.0/3.0)*f[gpu_fieldn_index(x-1, y, 15)] - (1.0/2.0)*f[gpu_fieldn_index(x-2, y, 15)];
	}

}

__device__ void device_outlet_FD(unsigned int x, unsigned int y, double *f){

	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = f[gpu_fieldn_index(x-3, y, n)];
	}
}

__device__ void device_outlet_FDP(unsigned int x, unsigned int y, double rho, double *f){
	
	double sumRho = 0.0;
	for(int n = 0; n < q; ++n){
		sumRho += f[gpu_fieldn_index(x-3, y, n)];
	}
	
	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = (rho/sumRho)*f[gpu_fieldn_index(x-3, y, n)];	
	}
}

__device__ void device_wall_velocity(unsigned int x, unsigned int y, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double moments[2];
	double ux = 0.0, uy = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	// South wall
	if(y == 0){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 3, 4, 7, 8, 11, 12, 13, 15, 16};

		known_moments(x, y, NI, I, f, moments);

		double rhoI = moments[0];
		double rhoaxy = moments[1];

		rho = 1.22336751176558*rhoI;
		tauxy = 2*rhoaxy;

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		gpu_recursive(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);
		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
		}
	}

	else if(y == 1){
		f[gpu_fieldn_index(x, y, 9)] = f[gpu_fieldn_index(x, y, 11)] - feq[gpu_fieldn_index(x, y, 11)] + feq[gpu_fieldn_index(x, y, 9)];
		f[gpu_fieldn_index(x, y, 10)] = f[gpu_fieldn_index(x, y, 12)] - feq[gpu_fieldn_index(x, y, 12)] + feq[gpu_fieldn_index(x, y, 10)];

		if(x > 0 && x < Nx_d-2){
			f[gpu_fieldn_index(x, y, 10)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y-1, 10)] + f[gpu_fieldn_index(x-1, y+1, 10)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y+2, 10)];
		}

		if(x > 1 && x < Nx_d-1){
			f[gpu_fieldn_index(x, y, 9)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y-1, 9)] + f[gpu_fieldn_index(x+1, y+1, 9)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y+2, 9)];
		}

		f[gpu_fieldn_index(x, y, 14)] = (1.0/2.0)*f[gpu_fieldn_index(x, y-1, 14)] + f[gpu_fieldn_index(x, y+2, 14)] - (1.0/2.0)*f[gpu_fieldn_index(x, y+3, 14)];
	}

	else if(y == 2){
		f[gpu_fieldn_index(x, y, 14)] = (1.0/6.0)*f[gpu_fieldn_index(x, y-2, 14)] + (4.0/3.0)*f[gpu_fieldn_index(x, y+1, 14)] - (1.0/2.0)*f[gpu_fieldn_index(x, y+2, 14)];
	}

	// North wall
	if(y == Ny_d-1){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 15};

		known_moments(x, y, NI, I, f, moments);

		double rhoI = moments[0];
		double rhoaxy = moments[1];

		rho = 1.22336751176558*rhoI;
		tauxy = 2*rhoaxy;

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		gpu_recursive(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);
		for(int n = 0; n < q; ++n){
			f[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
		}
	}

	else if(y == Ny_d-2){
		f[gpu_fieldn_index(x, y, 11)] = f[gpu_fieldn_index(x, y, 9)] - feq[gpu_fieldn_index(x, y, 9)] + feq[gpu_fieldn_index(x, y, 11)];
		f[gpu_fieldn_index(x, y, 12)] = f[gpu_fieldn_index(x, y, 10)] - feq[gpu_fieldn_index(x, y, 10)] + feq[gpu_fieldn_index(x, y, 12)];

		if(x > 0 && x < Nx_d-2){
			f[gpu_fieldn_index(x, y, 12)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y+1, 12)] + f[gpu_fieldn_index(x+1, y-1, 12)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y-2, 12)];
		}

		if(x > 1 && x < Nx_d-1){
			f[gpu_fieldn_index(x, y, 11)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y+1, 11)] + f[gpu_fieldn_index(x-1, y-1, 11)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y-2, 11)];
		}

		f[gpu_fieldn_index(x, y, 16)] = (1.0/2.0)*f[gpu_fieldn_index(x, y+1, 16)] + f[gpu_fieldn_index(x, y-2, 16)] - (1.0/2.0)*f[gpu_fieldn_index(x, y-3, 16)];
	}

	else if(y == Ny_d-3){
		f[gpu_fieldn_index(x, y, 16)] = (1.0/6.0)*f[gpu_fieldn_index(x, y+2, 16)] + (4.0/3.0)*f[gpu_fieldn_index(x, y-1, 16)] - (1.0/2.0)*f[gpu_fieldn_index(x, y-2, 16)];
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

__host__ void inlet_BC(double rho, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, std::string mode){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	unsigned int mode_num;
	if(mode == "VP"){
		mode_num = 1;
	}
	else if(mode == "PP"){
		mode_num = 2;
	}



	gpu_inlet<<< grid, block >>>(ux_in, f, feq, frec, r, u, v, txx, txy, tyy, mode_num);
	getLastCudaError("gpu_inlet kernel error");
}

__global__ void gpu_inlet(double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_inlet = inlet_d[gpu_scalar_index(x, y)];
	if(node_inlet){
		if(mode_num == 1){
			device_inlet_VP(x, y, ux_in, f, feq, frec, r, u, v, txx, txy, tyy);
		}
		else if(mode_num == 2){

		}
	}

	__syncthreads();
	att_moments(x, y, f, r, u, v, txx, txy, tyy);
}

__host__ void outlet_BC(double rho, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, std::string mode){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	unsigned int mode_num;
	if(mode == "FD"){
		mode_num = 1;
	}
	else if(mode == "FDP"){
		mode_num = 2;
	}
	else if(mode == "VP"){
		mode_num = 3;
	}
	else if(mode == "PP"){
		mode_num = 4
	}

	gpu_outlet<<< grid, block >>>(rho, ux_in, f, feq, frec, r, u, v, txx, txy, tyy, mode_num);
	getLastCudaError("gpu_outlet kernel error");
}

__global__ void gpu_outlet(double rho, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_outlet = outlet_d[gpu_scalar_index(x, y)];
	if(node_outlet){
		if(mode_num == 1){
			device_outlet_FD(x, y, f);
		}
		else if(mode_num == 2){
			device_outlet_FDP(x, y, rho, f);
		}
		else if(mode_num == 3){
			device_outlet_VP(x, y, ux_in, f, feq, frec, r, u, v, txx, txy, tyy);
		}
		else if(mode_num == 4){

		}
	}

	__syncthreads();
	att_moments(x, y, f, r, u, v, txx, txy, tyy);
}

__host__ void wall_velocity(double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_wall_velocity<<< grid, block >>>(f, feq, frec, r, u, v, txx, txy, tyy);
	getLastCudaError("gpu_wall_velocity kernel error");
}

__global__ void gpu_wall_velocity(double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_wall = walls_d[gpu_scalar_index(x, y)];
	if(node_wall){
		device_wall_velocity(x, y, f, feq, frec, r, u, v, txx, txy, tyy);
	}

	__syncthreads();
	att_moments(x, y, f, r, u, v, txx, txy, tyy);
}