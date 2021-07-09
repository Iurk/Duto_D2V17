#include "LBM.h"
#include "dados.h"
#include "boundary.h"
#include "system_solve.h"

using namespace myGlobals;

__host__ __device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	#if defined(__CUDA_ARCH__)
		return Nx_d*y + x;
	#else
		return Nx*y + x;
	#endif
}

__host__ __device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	#if defined(__CUDA_ARCH__)
		return Nx_d*(Ny_d*(d) + y) + x;
	#else
		return Nx*(Ny*(d) + y) + x;
	#endif
}

typedef struct{
	unsigned int x, y, NI;
	unsigned int *I, *IN;
	double *f;
	double rho, ux, uy;
}*InputData;

__host__ int func_velocity(N_Vector, N_Vector, void*);
__host__ void inlet_VP(double, double*);
__host__ void host_VP(unsigned int, unsigned int*, unsigned int*, double, double, double*, std::string);
__host__ void host_recursive(unsigned int, unsigned int, double, double, double, double, double, double, double*);
__host__ double host_recursive_n(unsigned int, unsigned int, unsigned int, double, double, double, double, double, double);



__host__ void host_inlet_PP(double, double*, double*);


__global__ void gpu_inlet(double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, unsigned int);
__global__ void gpu_bounce_back(double*);
__global__ void gpu_outlet(double, double, double*, double*, double*, double*, double*, double*, double*, double*, double*, unsigned int);
__global__ void gpu_wall_velocity(double*, double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void gpu_corners(double*, double*, double*, double*, double*, double*, double*, double*);

// Distribution
__device__ void gpu_recursive_inlet_pressure(unsigned int x, unsigned int y, double rho, double *a, double *frec){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	// Calculating the regularized recursive distribution
	double H[10];
	for(int n = 0; n < q; ++n){
		hermite_polynomial(ex_d[n], ey_d[n], cs, H);

		double order_1 = A*(a[1]*H[1] + a[2]*H[2]);
		double order_2 = B*(a[3]*H[3] + 2*a[4]*H[4] + a[5]*H[5]);
		double order_3 = C*(a[6]*H[6] + 3*a[7]*H[7] + 3*a[8]*H[8] + a[9]*H[9]);

		frec[gpu_fieldn_index(x, y, n)] = W[n]*(a[0]*H[0] + order_1 + order_2 + order_3);
	}
}

// Moments
__device__ void known_moments(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *rhoI){

	double rhoI_local = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		rhoI_local += f[gpu_fieldn_index(x, y, ni)];
	}

	*rhoI = rhoI_local;
}

__device__ void known_moments(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *rhoI, double *rhomxy){

	double rhoI_local = 0.0, rhomxy_local = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		rhoI_local += f[gpu_fieldn_index(x, y, ni)];
		rhomxy_local += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
	}
	
	*rhoI = rhoI_local;
	*rhomxy = rhomxy_local;
}

__device__ void known_moments(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *rhoI, double *rhomxx, double *rhomxy, double *rhomxxx, double *rhomxxy){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;

	double rhoI_local = 0.0, rhomxx_local = 0.0, rhomxy_local = 0.0, rhomxxx_local = 0.0, rhomxxy_local = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		double ex2 = ex_d[ni]*ex_d[ni];
		double ey2 = ey_d[ni]*ey_d[ni];

		rhoI_local += f[gpu_fieldn_index(x, y, ni)];
		rhomxx_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2);
		rhomxy_local += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
		rhomxxx_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - 3*cs2)*ex_d[ni];
		rhomxxy_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2)*ey_d[ni];
	}

	*rhoI = rhoI_local;
	*rhomxx = rhomxx_local;
	*rhomxy = rhomxy_local;
	*rhomxxx = rhomxxx_local;
	*rhomxxy = rhomxxy_local;
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

	double uy_in = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	ux_in = poiseulle_eval(x, y);

	double rhoI = 0, rhomxy = 0;
	double *prhoI = &rhoI, *prhomxy = &rhomxy;

	if(x == 0){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};

		known_moments(x, y, NI, I, f, prhoI, prhomxy);

		double rhoI = *prhoI;
		double rhomxy = *prhomxy;

		double ux_in2 = ux_in*ux_in;
		double ux_in3 = ux_in*ux_in*ux_in;

		rho = (129600*rhoI)/(5594.46221994725*ux_in3 - 55552.8525416427*ux_in2 - 64146.5596724039*ux_in + 105937.094743475);
		tauxy = (-270*rhomxy)/(245.569775957799*ux_in - 135);

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux_in;
		v[gpu_scalar_index(x, y)] = uy_in;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		recursive_dist(x, y, rho, ux_in, uy_in, tauxx, tauxy, tauyy, frec);
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

__device__ void device__inlet_interpolation(unsigned int x, unsigned int y, double *f, double *feq){

	if(x == 1){
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

__device__ void device_outlet_interpolation(unsigned int x, unsigned int y, double *f, double *feq){

	if(x == Nx_d-2){
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

__device__ void device_outlet_VP(unsigned int x, unsigned int y, double ux_out, double *f, double *feq, double *frec, double *r, double *u, double *v, double*txx, double *txy, double *tyy){

	double uy_out = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	ux_out = poiseulle_eval(x, y);

	double rhoI = 0, rhomxy = 0;
	double *prhoI = &rhoI, *prhomxy = &rhomxy;

	if(x == Nx_d-1){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 2, 4, 5, 8, 9, 12, 13, 14, 16};

		known_moments(x, y, NI, I, f, prhoI, prhomxy);

		rhoI = *prhoI;
		rhomxy = *prhomxy;

		double ux_out2 = ux_out*ux_out;
		double ux_out3 = ux_out*ux_out*ux_out;

		rho = (-129600*rhoI)/(5594.46221994725*ux_out3 + 55552.8525416427*ux_out2 - 64146.5596724039*ux_out - 105937.094743475);
		tauxy = (270*rhomxy)/(245.569775957799*ux_out + 135);

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux_out;
		v[gpu_scalar_index(x, y)] = uy_out;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		recursive_dist(x, y, rho, ux_out, uy_out, tauxx, tauxy, tauyy, frec);
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

__device__ void device_outlet_FDP(unsigned int x, unsigned int y, double rho_out, double *f){
	
	double sumRho = 0.0;
	for(int n = 0; n < q; ++n){
		sumRho += f[gpu_fieldn_index(x-3, y, n)];
	}
	
	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = (rho_out/sumRho)*f[gpu_fieldn_index(x-3, y, n)];	
	}
}

__device__ void device_wall_velocity(unsigned int x, unsigned int y, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double ux = 0.0, uy = 0.0;
	double tauxx = 0.0, tauyy = 0.0;

	double rho, tauxy;

	double rhoI = 0, rhomxy = 0;
	double *prhoI = &rhoI, *prhomxy = &rhomxy;

	// South wall
	if(y == 0){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 3, 4, 7, 8, 11, 12, 13, 15, 16};

		known_moments(x, y, NI, I, f, prhoI, prhomxy);

		rhoI = *prhoI;
		rhomxy = *prhomxy;

		rho = 1.22336751176558*rhoI;
		tauxy = 2*rhomxy;

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		recursive_dist(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);
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

		known_moments(x, y, NI, I, f, prhoI, prhomxy);

		rhoI = *prhoI;
		rhomxy = *prhomxy;

		rho = 1.22336751176558*rhoI;
		tauxy = 2*rhomxy;

		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		recursive_dist(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);
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

__device__ void device_corners_first(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *frec){

	double ux = 0.0, uy = 0.0;
	double tauxx = 0.0, tauxy = 0.0, tauyy = 0.0;

	double rhoI = 0;
	double *prhoI = &rhoI;
	known_moments(x, y, NI, I, f, prhoI);

	rhoI = *prhoI;
	double rho = 1.4971916997958*rhoI;

	recursive_dist(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);

	for(int n = 0; n < q; ++n){
		f[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
	}
}

__device__ void device_corners_others(unsigned int x, unsigned int y, double *f){

	if(x == 0){
		x += 1;

		if(y == 0){
			y += 1;

			f[gpu_fieldn_index(x, y, 9)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y-1, 9)] + f[gpu_fieldn_index(x+1, y+1, 9)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y+2, 9)];
			f[gpu_fieldn_index(x, y, 13)] = 0.5*f[gpu_fieldn_index(x-1, y, 13)] + f[gpu_fieldn_index(x+2, y, 13)] - 0.5*f[gpu_fieldn_index(x+3, y, 13)];
			f[gpu_fieldn_index(x, y, 14)] = 0.5*f[gpu_fieldn_index(x, y-1, 14)] + f[gpu_fieldn_index(x, y+2, 14)] - 0.5*f[gpu_fieldn_index(x, y+3, 14)];
			f[gpu_fieldn_index(x, y, 10)] = 0.5*(f[gpu_fieldn_index(x-1, y+1, 10)] + f[gpu_fieldn_index(x+1, y-1, 10)]);
			f[gpu_fieldn_index(x, y, 12)] = 0.5*(f[gpu_fieldn_index(x-1, y+1, 12)] + f[gpu_fieldn_index(x+1, y-1, 12)]);

			x += 1;
			y += 1;
			f[gpu_fieldn_index(x, y, 13)] = (1.0/6.0)*f[gpu_fieldn_index(x-2, y, 13)] + (4.0/3.0)*f[gpu_fieldn_index(x+1, y, 13)] - 0.5*f[gpu_fieldn_index(x+2, y, 13)];
			f[gpu_fieldn_index(x, y, 14)] = (1.0/6.0)*f[gpu_fieldn_index(x, y-2, 14)] + (4.0/3.0)*f[gpu_fieldn_index(x, y+1, 14)] - 0.5*f[gpu_fieldn_index(x, y+2, 14)];
		}

		else if(y == Ny_d-1){
			y -= 1;

			f[gpu_fieldn_index(x, y, 12)] = (1.0/3.0)*f[gpu_fieldn_index(x-1, y+1, 12)] + f[gpu_fieldn_index(x+1, y-1, 12)] - (1.0/3.0)*f[gpu_fieldn_index(x+2, y-2, 12)];
			f[gpu_fieldn_index(x, y, 13)] = 0.5*f[gpu_fieldn_index(x-1, y, 13)] + f[gpu_fieldn_index(x+2, y, 13)] - 0.5*f[gpu_fieldn_index(x+3, y, 13)];
			f[gpu_fieldn_index(x, y, 16)] = 0.5*f[gpu_fieldn_index(x, y+1, 16)] + f[gpu_fieldn_index(x, y-2, 16)] - 0.5*f[gpu_fieldn_index(x, y-3, 16)];
			f[gpu_fieldn_index(x, y, 9)] = 0.5*(f[gpu_fieldn_index(x-1, y-1, 9)] + f[gpu_fieldn_index(x+1, y+1, 9)]);
			f[gpu_fieldn_index(x, y, 11)] = 0.5*(f[gpu_fieldn_index(x-1, y-1, 11)] + f[gpu_fieldn_index(x+1, y+1, 11)]);

			x += 1;
			y -= 1;
			f[gpu_fieldn_index(x, y, 13)] = (1.0/6.0)*f[gpu_fieldn_index(x-2, y, 13)] + (4.0/3.0)*f[gpu_fieldn_index(x+1, y, 13)] - 0.5*f[gpu_fieldn_index(x+2, y, 13)];
			f[gpu_fieldn_index(x, y, 14)] = (1.0/6.0)*f[gpu_fieldn_index(x, y+2, 14)] + (4.0/3.0)*f[gpu_fieldn_index(x, y-1, 14)] - 0.5*f[gpu_fieldn_index(x, y-2, 14)];
		}
	}

	if(x == Nx_d-1){
		x -= 1;

		if(y == 0){
			y+= 1;

			f[gpu_fieldn_index(x, y, 10)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y-1, 10)] + f[gpu_fieldn_index(x-1, y+1, 10)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y+2, 10)];
			f[gpu_fieldn_index(x, y, 14)] = 0.5*f[gpu_fieldn_index(x, y-1, 14)] + f[gpu_fieldn_index(x, y+2, 14)] - 0.5*f[gpu_fieldn_index(x, y+3, 14)];
			f[gpu_fieldn_index(x, y, 15)] = 0.5*f[gpu_fieldn_index(x+1, y, 15)] + f[gpu_fieldn_index(x-2, y, 15)] - 0.5*f[gpu_fieldn_index(x-3, y, 15)];
			f[gpu_fieldn_index(x, y, 9)] = 0.5*(f[gpu_fieldn_index(x-1, y-1, 9)] + f[gpu_fieldn_index(x+1, y+1, 9)]);
			f[gpu_fieldn_index(x, y, 11)] = 0.5*(f[gpu_fieldn_index(x-1, y-1, 11)] + f[gpu_fieldn_index(x+1, y+1, 11)]);

			x -= 1;
			y += 1;
			f[gpu_fieldn_index(x, y, 14)] = (1.0/6.0)*f[gpu_fieldn_index(x, y-2, 14)] + (4.0/3.0)*f[gpu_fieldn_index(x, y+1, 14)] - 0.5*f[gpu_fieldn_index(x, y+2, 14)];
			f[gpu_fieldn_index(x, y, 15)] = (1.0/6.0)*f[gpu_fieldn_index(x+2, y, 15)] + (4.0/3.0)*f[gpu_fieldn_index(x-1, y, 15)] - 0.5*f[gpu_fieldn_index(x-2, y, 15)];
		}

		else if(y == Ny_d-1){
			y -= 1;

			f[gpu_fieldn_index(x, y, 11)] = (1.0/3.0)*f[gpu_fieldn_index(x+1, y+1, 11)] + f[gpu_fieldn_index(x-1, y-1, 11)] - (1.0/3.0)*f[gpu_fieldn_index(x-2, y-2, 11)];
			f[gpu_fieldn_index(x, y, 15)] = 0.5*f[gpu_fieldn_index(x+1, y, 15)] + f[gpu_fieldn_index(x-2, y, 15)] - 0.5*f[gpu_fieldn_index(x-3, y, 15)];
			f[gpu_fieldn_index(x, y, 16)] = 0.5*f[gpu_fieldn_index(x, y+1, 16)] + f[gpu_fieldn_index(x, y-2, 16)] - 0.5*f[gpu_fieldn_index(x, y-3, 16)];
			f[gpu_fieldn_index(x, y, 10)] = 0.5*(f[gpu_fieldn_index(x-1, y+1, 10)] + f[gpu_fieldn_index(x+1, y-1, 10)]);
			f[gpu_fieldn_index(x, y, 12)] = 0.5*(f[gpu_fieldn_index(x-1, y+1, 12)] + f[gpu_fieldn_index(x+1, y-1, 12)]);

			x -= 1;
			y -= 1;
			f[gpu_fieldn_index(x, y, 15)] = (1.0/6.0)*f[gpu_fieldn_index(x+2, y, 15)] + (4.0/3.0)*f[gpu_fieldn_index(x-1, y, 15)] - 0.5*f[gpu_fieldn_index(x-2, y, 15)];
			f[gpu_fieldn_index(x, y, 16)] = (1.0/6.0)*f[gpu_fieldn_index(x, y+2, 16)] + (4.0/3.0)*f[gpu_fieldn_index(x, y-1, 16)] - 0.5*f[gpu_fieldn_index(x, y-2, 16)];
		}
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

__host__ void inlet_BC(double rho_in, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, std::string mode){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	unsigned int mode_num;
	if(mode == "VP"){
		mode_num = 1;
		inlet_VP(ux_in, f);
	}
	else if(mode == "PP"){
		mode_num = 2;
		//host_inlet_PP(rho_in, solution, f);
	}

	//printf("Hi before calling the kernel\n");
	//printf("mode_num: %d\n", mode_num);
	gpu_inlet<<< grid, block >>>(rho_in, ux_in, f, feq, frec, r, u, v, txx, txy, tyy, mode_num);
	getLastCudaError("gpu_inlet kernel error");
}

__host__ void inlet_VP(double ux_in, double *f_gpu){

	double uy_in = 0.0;

	unsigned int NI = 11;
	unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};
	unsigned int IN[6] = {1, 5, 8, 9, 12, 13};

	host_VP(NI, I, IN, ux_in, uy_in, f_gpu, "inlet");
}

__host__ void outlet_VP(double ux_in, double *f_gpu){

	double uy_in = 0.0;

	unsigned int NI = 11;
	unsigned int I[11] = {0, 1, 2, 4, 5, 8, 9, 12, 13, 14, 16};
	unsigned int IN[6] = {3, 6, 7, 10, 11, 15};

	host_VP(NI, I, IN, ux_in, uy_in, f_gpu, "outlet");
}

__host__ void host_VP(unsigned int NI, unsigned int *I, unsigned int *IN, double ux, double uy, double *f_gpu, std::string mode){

	double solution[2];
	double guess[2] = {0.0, 0.0};

	double *f_pinned;
	checkCudaErrors(cudaMallocHost((void**)&f_pinned, mem_size_ndir));
	checkCudaErrors(cudaMemcpy(f_pinned, f_gpu, mem_size_ndir, cudaMemcpyDeviceToHost));

	InputData input = NULL;
	input = (InputData)malloc(sizeof *input);
	input->NI = NI;
	input->I = I;
	input->IN = IN;
	input->f = f_pinned;
	input->uy = uy;

	unsigned int x;
	if(mode == "inlet"){
		x = 0;
		input->x = x;
	}
	else if(mode == "outlet"){
		x = Nx-1;
		input->x = x;
	}

	for(unsigned int y = 0; y < Ny; ++y){
		input->y = y;

		ux = poiseulle_eval(x, y);
		input-> ux = ux;

		solving(2, guess, solution, input, func_velocity);

		double rho = solution[0];
		double tauxy = solution[1];
		host_recursive(x, y, rho, ux, uy, 0.0, tauxy, 0.0, f_pinned);
	}

	checkCudaErrors(cudaMemcpy(f_gpu, f_pinned, mem_size_ndir, cudaMemcpyHostToDevice));
	free(input);
}
/*
__host__ void host_inlet_PP(double rho_in, double *solution, double *f_gpu){

	double uy_in = 0.0;

	double guess[NVAR] = {0.0, 0.0, 0.0, 0.0};

	unsigned int NI = 11;
	unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};
	unsigned int IN[6] = {1, 5, 8, 9, 12, 13};

	double *f_pinned;
	checkCudaErrors(cudaMallocHost((void**)&f_pinned, mem_size_ndir));
	checkCudaErrors(cudaMemcpy(f_pinned, f_gpu, mem_size_ndir, cudaMemcpyDeviceToHost));

	InputData input = NULL;
	input = (InputData)malloc(sizeof *input);
	input->NI = NI;
	input->I = I;
	input->IN = IN;
	input->f = f_pinned;
	input->rho = rho_in;
	input->uy = uy_in;

	for(int y = 0; y < Ny; ++y){
		int x = 0;

		input->x = x;
		input->y = y;
		solving(guess, solution, input, func);

		double ux = solution[0];
		double tauxx = solution[1];
		double tauxy = solution[2];
		double tauyy = solution[3];
		host_recursive(x, y, rho_in, ux, uy_in, tauxx, tauxy, tauyy, f_pinned);

		if(y == 5){
			printf("f_pinned\n");
			printf("f0: %g f1: %g f2: %g\n", f_pinned[gpu_fieldn_index(x, y, 0)], f_pinned[gpu_fieldn_index(x, y, 1)], f_pinned[gpu_fieldn_index(x, y, 2)]);
			printf("f3: %g f4: %g f5: %g\n", f_pinned[gpu_fieldn_index(x, y, 3)], f_pinned[gpu_fieldn_index(x, y, 4)], f_pinned[gpu_fieldn_index(x, y, 5)]);
			printf("f6: %g f7: %g f8: %g\n", f_pinned[gpu_fieldn_index(x, y, 6)], f_pinned[gpu_fieldn_index(x, y, 7)], f_pinned[gpu_fieldn_index(x, y, 8)]);
			printf("f9: %g f10: %g f11: %g\n", f_pinned[gpu_fieldn_index(x, y, 9)], f_pinned[gpu_fieldn_index(x, y, 10)], f_pinned[gpu_fieldn_index(x, y, 11)]);
			printf("f12: %g f13: %g f14: %g\n", f_pinned[gpu_fieldn_index(x, y, 12)], f_pinned[gpu_fieldn_index(x, y, 13)], f_pinned[gpu_fieldn_index(x, y, 14)]);
			printf("f15: %g f16: %g\n", f_pinned[gpu_fieldn_index(x, y, 15)], f_pinned[gpu_fieldn_index(x, y, 16)]);
		}
	}

	checkCudaErrors(cudaMemcpy(f_gpu, f_pinned, mem_size_ndir, cudaMemcpyHostToDevice));
	printf("Copied\n");
}
*/
__global__ void gpu_inlet(double rho_in, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_inlet = inlet_d[gpu_scalar_index(x, y)];
	if(node_inlet){
		if(mode_num == 1){
			//device_inlet_VP(x, y, ux_in, f, feq, frec, r, u, v, txx, txy, tyy);
			device__inlet_interpolation(x, y, f, feq);
		}
		else if(mode_num == 2){
			//device_inlet_PP(x, y, rho_in, f, feq, frec, r, u, v, txx, txy, tyy);
			//device_interpolation(x, y, f, feq);
		}
	}
}

__host__ void outlet_BC(double rho_out, double ux_out, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, std::string mode){

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
		outlet_VP(ux_out, f);
	}
	else if(mode == "PP"){
		mode_num = 4;
	}

	gpu_outlet<<< grid, block >>>(rho_out, ux_out, f, feq, frec, r, u, v, txx, txy, tyy, mode_num);
	getLastCudaError("gpu_outlet kernel error");
}

__global__ void gpu_outlet(double rho_out, double ux_out, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_outlet = outlet_d[gpu_scalar_index(x, y)];
	if(node_outlet){
		if(mode_num == 1){
			device_outlet_FD(x, y, f);
		}
		else if(mode_num == 2){
			device_outlet_FDP(x, y, rho_out, f);
		}
		else if(mode_num == 3){
			//device_outlet_VP(x, y, ux_out, f, feq, frec, r, u, v, txx, txy, tyy);
			device_outlet_interpolation(x, y, f, feq);
		}
		else if(mode_num == 4){
			//device_outlet_PP(x, y, rho_out, f, feq, frec, r, u, v, txx, txy, tyy);
		}
	}
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
}

__host__ void corners(double *f, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	dim3 grid(1, 1, 1);
	dim3 block(1, 1, 1);

	gpu_corners<<< grid, block >>>(f, frec, r, u, v, txx, txy, tyy);
	getLastCudaError("gpu_corners kernel error");
}

__global__ void gpu_corners(double *f, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	unsigned int NI, I[7], x, y;
	// Southwest
	NI = 7;
	I[0] = 0, I[1] = 3, I[2] = 4, I[3] = 7, I[4] = 11, I[5] = 15, I[6] = 16;

	x = 0, y = 0;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	// Northwest
	NI = 7;
	I[0] = 0, I[1] = 2, I[2] = 3, I[3] = 6, I[4] = 10, I[5] = 14, I[6] = 15;

	x = 0, y = Ny_d-1;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	// Southeast
	NI = 7;
	I[0] = 0, I[1] = 1, I[2] = 4, I[3] = 8, I[4] = 12, I[5] = 13, I[6] = 16;

	x = Nx_d-1, y = 0;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	// Northeast
	NI = 7;
	I[0] = 0, I[1] = 1, I[2] = 2, I[3] = 5, I[4] = 9, I[5] = 13, I[6] = 14;

	x = Nx_d-1, y = Ny_d-1;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);
}

__host__ int func_velocity(N_Vector u, N_Vector f, void *user_data){

	realtype *udata, *fdata;
	realtype rho, ux, uy, tauxx, tauxy, tauyy;

	unsigned int x, y, NI;
	unsigned int *I, *IN;
	double *frec;

	InputData input = (InputData)user_data;

	x = input->x;
	y = input->y;
	NI = input->NI;
	I = input->I;
	IN = input->IN;
	frec = input->f;
	ux = input->ux;
	uy = input->uy;

	udata = N_VGetArrayPointer(u);
	fdata = N_VGetArrayPointer(f);

	rho = udata[0];
	tauxx = 0.0;
	tauxy = udata[1];
	tauyy = 0.0;

	double cs = 1.0/as;
	double cs2 = cs*cs;

	double rhoI = 0.0, rhoImxy = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		rhoI += frec[gpu_fieldn_index(x, y, ni)];
		rhoImxy += frec[gpu_fieldn_index(x, y, ni)]*ex[ni]*ey[ni];
	}

	double rhoIN = 0.0, rhoINmxy = 0.0;
	for(int n = 0; n < ndir-NI; ++n){
		unsigned int ni = IN[n];

		rhoIN += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy);
		rhoINmxy += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy)*ex[ni]*ey[ni];
	}

	fdata[0] = rhoI + rhoIN - rho;
	fdata[1] = rhoImxy + rhoINmxy - rho*ux*uy + tauxy;

	return (0);
}

__host__ int func_pressure(N_Vector u, N_Vector f, void *user_data){

	realtype *udata, *fdata;
	realtype ux, tauxx, tauxy, tauyy;

	unsigned int x, y, NI;
	unsigned int *I, *IN;
	double *frec;
	double rho, uy;

	InputData input = (InputData)user_data;

	x = input->x;
	y = input->y;
	NI = input->NI;
	I = input->I;
	IN = input->IN;
	frec = input->f;
	rho = input->rho;
	uy = input->uy;

	udata = N_VGetArrayPointer(u);
	fdata = N_VGetArrayPointer(f);

	ux = udata[0];
	tauxx = udata[1];
	tauxy = udata[2];
	tauyy = udata[3];

	double cs = 1.0/as;
	double cs2 = cs*cs;

	double rhoIax = 0.0, rhoIaxx = 0.0, rhoIaxy = 0.0, rhoIayy = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		double ex2 = ex[ni]*ex[ni];
		double ey2 = ey[ni]*ey[ni];

		rhoIax += frec[gpu_fieldn_index(x, y, ni)]*ex[ni];
		rhoIaxx += frec[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2);
		rhoIaxy += frec[gpu_fieldn_index(x, y, ni)]*ex[ni]*ey[ni];
		rhoIayy += frec[gpu_fieldn_index(x, y, ni)]*(ey2 - cs2);
	}

	double rhoINax = 0.0, rhoINaxx = 0.0, rhoINaxy = 0.0, rhoINayy = 0.0;
	for(int n = 0; n < ndir-NI; ++n){
		unsigned int ni = IN[n];

		double ex2 = ex[ni]*ex[ni];
		double ey2 = ey[ni]*ey[ni];

		rhoINax += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy)*ex[ni];
		rhoINaxx += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy)*(ex2 - cs2);
		rhoINaxy += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy)*ex[ni]*ey[ni];
		rhoINayy += host_recursive_n(ni, x, y, rho, ux, uy, tauxx, tauxy, tauyy)*(ey2 - cs2);
	}

	fdata[0] = rhoIax + rhoINax - rho*ux;
	fdata[1] = rhoIaxx + rhoINaxx - rho*ux*ux - tauxx;
	fdata[2] = rhoIaxy + rhoINaxy - rho*ux*uy - tauxy;
	fdata[3] = rhoIayy + rhoIayy - rho*uy*uy - tauyy;

	return (0);
}

__host__ double host_recursive_n(unsigned int n, unsigned int x, unsigned int y, double rho, double ux, double uy, double tauxx, double tauxy, double tauyy){

	double frec;

	double cs = 1.0/as;
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);

	double W[] = {w0, wp, wp, wp, wp, ws, ws, ws, ws, wt, wt, wt, wt, wq, wq, wq, wq};

	// Calculating the regularized recursive distribution
	double a[10], H[10];
	hermite_polynomial(ex[n], ey[n], cs, H);
	hermite_moments(rho, ux, uy, tauxx, tauxy, tauyy, a);

	double order_1 = A*(a[1]*H[1] + a[2]*H[2]);
	double order_2 = B*(a[3]*H[3] + 2*a[4]*H[4] + a[5]*H[5]);
	double order_3 = C*(a[6]*H[6] + 3*a[7]*H[7] + 3*a[8]*H[8] + a[9]*H[9]);

	frec = W[n]*(a[0]*H[0] + order_1 + order_2 + order_3);
	return frec;
}

// Recursive Regularized Distribuition
__host__ void host_recursive(unsigned int x, unsigned int y, double rho, double ux, double uy, double tauxx, double tauxy, double tauyy, double *frec){

	double cs = 1.0/as;
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);

	double W[] = {w0, wp, wp, wp, wp, ws, ws, ws, ws, wt, wt, wt, wt, wq, wq, wq, wq};

	// Calculating the regularized recursive distribution
	double a[10], H[10];
	for(int n = 0; n < ndir; ++n){
		hermite_polynomial(ex[n], ey[n], cs, H);
		hermite_moments(rho, ux, uy, tauxx, tauxy, tauyy, a);

		double order_1 = A*(a[1]*H[1] + a[2]*H[2]);
		double order_2 = B*(a[3]*H[3] + 2*a[4]*H[4] + a[5]*H[5]);
		double order_3 = C*(a[6]*H[6] + 3*a[7]*H[7] + 3*a[8]*H[8] + a[9]*H[9]);

		frec[gpu_fieldn_index(x, y, n)] = W[n]*(a[0]*H[0] + order_1 + order_2 + order_3);
	}
}