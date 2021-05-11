#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>

#include <errno.h>

#include "LBM.h"
#include "dados.h"
#include "boundary.h"

using namespace myGlobals;

// Input data
__constant__ unsigned int q, Nx_d, Ny_d;
__constant__ double rho0_d, u_max_d, nu_d, tau_d, mi_ar_d;

//Lattice Data
__constant__ double as_d, w0_d, wp_d, ws_d, wt_d, wq_d;
__device__ int *ex_d, *ey_d;

// Mesh data
__device__ bool *walls_d, *inlet_d, *outlet_d;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_init_equilibrium(double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_convergence(double*, double*, double*);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_print_mesh(int);
__global__ void gpu_initialization(double*, double);

// Hermite Polynomials
__device__ void hermite_polynomial(int ex, int ey, double cs, double *H){

	double cs2 = cs*cs;

	double ex2 = ex*ex;
	double ey2 = ey*ey;
	double ex3 = ex*ex*ex;
	double ey3 = ey*ey*ey;

	H[0] = 1;					// 0
	H[1] = ex;					// 1 - x
	H[2] = ey;					// 1 - y
	H[3] = ex2 - cs2;			// 2 - xx
	H[4] = ex*ey;				// 2 - xy
	H[5] = ey2 - cs2;			// 2 - yy
	H[6] = ex3 - 3*ex*cs2;		// 3 - xxx
	H[7] = ex2*ey - ey*cs2;		// 3 - yxx
	H[8] = ex*ey2 - ex*cs2;		// 3 - xyy
	H[9] = ey3 - 3*ey*cs2;		// 3 - yyy

}

// Hermite Moments
__device__ void hermite_moments(double rho, double ux, double uy, double tauxx, double tauxy, double tauyy, double *a){

	double ux2 = ux*ux;
	double uy2 = uy*uy;
	double ux3 = ux*ux*ux;
	double uy3 = uy*uy*uy;

	a[0] = rho;											// 0
	a[1] = rho*ux;										// 1 - x
	a[2] = rho*uy;										// 1 - y
	a[3] = rho*ux2 + tauxx;								// 2 - xx
	a[4] = rho*ux*uy + tauxy;							// 2 - xy
	a[5] = rho*uy2 + tauyy;								// 2 - yy
	a[6] = rho*ux3 + 3*ux*tauxx;						// 3 - xxx
	a[7] = rho*ux2*uy + 2*ux*tauxy + uy*tauxx;			// 3 - yxx
	a[8] = rho*ux*uy2 + 2*uy*tauxy + ux*tauyy;			// 3 - xyy
	a[9] = rho*uy3 + 3*uy*tauyy;						// 3 - yyy
}

// Equilibrium Distribuition
__device__ void gpu_equilibrium(unsigned int x, unsigned int y, double rho, double ux, double uy, double *feq){

	double cs = 1.0/as_d;
	
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	for(int n = 0; n < q; ++n){

		double ux2 = ux*ux;
		double uy2 = uy*uy;
		double ux3 = ux*ux*ux;
		double uy3 = uy*uy*uy;

		double ex2 = ex_d[n]*ex_d[n];
		double ey2 = ey_d[n]*ey_d[n];
		double ex3 = ex_d[n]*ex_d[n]*ex_d[n];
		double ey3 = ey_d[n]*ey_d[n]*ey_d[n];

		double order_1 = A*(ux*ex_d[n] + uy*ey_d[n]);
		double order_2 = B*(ux2*(ex2 - cs2) + 2*ux*uy*ex_d[n]*ey_d[n] + uy2*(ey2 - cs2));
		
		double xxx = ux3*(ex3 - 3*ex_d[n]*cs2);
		double yxx = ux2*uy*(ex2*ey_d[n] - ey_d[n]*cs2);
		double xyy = ux*uy2*(ex_d[n]*ey2 - ex_d[n]*cs2);
		double yyy = uy3*(ey3 - 3*ey_d[n]*cs2);
		double order_3 = C*(xxx + 3*yxx + 3*xyy + yyy);

		feq[gpu_fieldn_index(x, y, n)] = W[n]*rho*(1 + order_1 + order_2 + order_3);
	}
}

// Recursive Regularized Distribuition
__device__ void gpu_recursive(unsigned int x, unsigned int y, double rho, double ux, double uy, double tauxx, double tauxy, double tauyy, double *frec){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	// Calculating the regularized recursive distribution
	double a[10], H[10];
	for(int n = 0; n < q; ++n){
		hermite_polynomial(ex_d[n], ey_d[n], cs, H);
		hermite_moments(rho, ux, uy, tauxx, tauxy, tauyy, a);

		double order_1 = A*(a[1]*H[1] + a[2]*H[2]);
		double order_2 = B*(a[3]*H[3] + 2*a[4]*H[4] + a[5]*H[5]);
		double order_3 = C*(a[6]*H[6] + 3*a[7]*H[7] + 3*a[8]*H[8] + a[9]*H[9]);

		frec[gpu_fieldn_index(x, y, n)] = W[n]*(a[0]*H[0] + order_1 + order_2 + order_3);
	}
}

__device__ void gpu_recursive_inlet_pressure(unsigned int x, unsigned int y, double *a, double *frec){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;
	double cs4 = cs2*cs2;
	double cs6 = cs4*cs2;

	double A = 1.0/(cs2);
	double B = 1.0/(2.0*cs4);
	double C = 1.0/(6.0*cs6);
/*
	if(x == 0){
		if(y == 5){
			printf("a0: %g ax: %g ay: %g axx: %g axy: %g ayy: %g\n", a[0], a[1], a[2], a[3], a[4], a[5]);
			printf("axxx: %g axxy: %g axyy: %g ayyy: %g\n", a[6], a[7], a[8], a[9]);
		}
	}
*/
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

__device__ void gpu_source(unsigned int x, unsigned int y, double gx, double gy, double rho, double ux, double uy, double *S){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;

	double A = 1.0/(cs2);
	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	for(int n = 0; n < q; ++n){
		double gdotei = gx*ex_d[n] + gy*ey_d[n];
		double udotei = ux*ex_d[n] + uy*ey_d[n];

		double order_1 = gx*(ex_d[n] - ux) + gy*(ey_d[n] - uy);
		double order_2 = A*gdotei*udotei;

		S[gpu_fieldn_index(x, y, n)] = A*W[n]*rho*(order_1 + order_2);
	}
}

// Poiseulle Flow
__device__ double poiseulle_eval(unsigned int x, unsigned int y, double *u){

	double y_double = (double) y;
	double Ny_double = (double) Ny_d;

	double gradP = (-1)*8*u_max_d*mi_ar_d/(Ny_double*Ny_double - 2*Ny_double);

	double ux = (1.0/(2.0*mi_ar_d))*(gradP)*(y_double*y_double - (Ny_double - 1)*y_double);

	return ux;
}

__host__ void init_equilibrium(double *f1, double *r, double *u, double *v){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_init_equilibrium<<< grid, block >>>(f1, r, u, v);
	getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f1, double *r, double *u, double *v){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];
	
	gpu_equilibrium(x, y, rho, ux, uy, f1);
}

__host__ void stream_collide_save(double *f1, double *f2, double *feq, double *frec, double *S, double *r, double *u, double *v, double *txx, double *txy, double *tyy, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	//dim3 grid(1,1,1);
	//dim3 block(1,1,1);

	gpu_stream_collide_save<<< grid, block >>>(f1, f2, feq, frec, S, r, u, v, txx, txy, tyy, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f1, double *f2, double *feq, double *frec, double *S, double *r, double *u, double *v, double *txx, double *txy, double *tyy, bool save){

	const double omega = 1.0/tau_d;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int x_att, y_att;

	double rho = 0, ux_i = 0, uy_i = 0, Pxx = 0, Pxy = 0, Pyy = 0;
	for(int n = 0; n < q; ++n){
		x_att = (x - ex_d[n] + Nx_d)%Nx_d;
		y_att = (y - ey_d[n] + Ny_d)%Ny_d;

		rho += f1[gpu_fieldn_index(x_att, y_att, n)];
		ux_i += (f1[gpu_fieldn_index(x_att, y_att, n)]*ex_d[n]);
		uy_i += (f1[gpu_fieldn_index(x_att, y_att, n)]*ey_d[n]);
		Pxx += f1[gpu_fieldn_index(x_att, y_att, n)]*ex_d[n]*ex_d[n];
		Pxy += f1[gpu_fieldn_index(x_att, y_att, n)]*ex_d[n]*ey_d[n];
		Pyy += f1[gpu_fieldn_index(x_att, y_att, n)]*ey_d[n]*ey_d[n];
	}

	double ux = ux_i/rho;
	double uy = uy_i/rho;

	double cs = 1.0/as_d;
	double cs2 = cs*cs;

	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;

	//gpu_source(x, y, gx, gy, rho, ux, uy, S);
	gpu_equilibrium(x, y, rho, ux, uy, feq);

	double fneq;
	double tauxx = 0.0, tauxy = 0.0, tauyy = 0.0;
	for(int n = 0; n < q; ++n){
		x_att = (x - ex_d[n] + Nx_d)%Nx_d;
		y_att = (y - ey_d[n] + Ny_d)%Ny_d;

		fneq = f1[gpu_fieldn_index(x_att, y_att, n)] - feq[gpu_fieldn_index(x, y, n)];

		tauxx += (fneq)*ex_d[n]*ex_d[n];
		tauxy += (fneq)*ex_d[n]*ey_d[n];
		tauyy += (fneq)*ey_d[n]*ey_d[n];
	}

	txx[gpu_scalar_index(x, y)] = tauxx;
	txy[gpu_scalar_index(x, y)] = tauxy;
	tyy[gpu_scalar_index(x, y)] = tauyy;
	
	gpu_recursive(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);

	// Collision Step
	for(int n = 0; n < q; ++n){
		f2[gpu_fieldn_index(x, y, n)] = omega*feq[gpu_fieldn_index(x, y, n)] + (1.0 - omega)*frec[gpu_fieldn_index(x, y, n)];
	}

	if(x == 0){
		double rho_in = 1.0;
		double uy_in = 0.0;

		unsigned int NI = 11;
		unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};

		double rhoax = 0.0, rhoaxx = 0.0, rhoaxy = 0.0, rhoaxxx = 0.0, rhoaxxy = 0.0;
		for(int n = 0; n < NI; ++n){
			unsigned int ni = I[n];

			double ex2 = ex_d[ni]*ex_d[ni];
			rhoax += f2[gpu_fieldn_index(x, y, ni)]*ex_d[n];
			rhoaxx += f2[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2);
			rhoaxy += f2[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
			rhoaxxx += f2[gpu_fieldn_index(x, y, ni)]*(ex2 - 3*cs2)*ex_d[n];
			rhoaxxy += f2[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2)*ey_d[n];
		}

		double ax = 0.801965942378094*rhoaxxx + 2.81424743597281*rhoaxx + 3.86310204316732*rhoax + 1.02814710445382*rho_in;
		double axx = 0.89703381557917*rhoaxxx + 3.14785825940274*rhoaxx + 2.08396073484102*rhoax + 0.740060658742184*rho_in;
		double axy = 2.57129750702885*rhoaxxy + 3.73177207142366*rhoaxy;
		double axxx = 2.28393633549532*rhoaxxx + 0.996385334990318*rhoaxx + 0.659631960457206*rhoax + 0.207957767581293*rho_in;
		double axxy = 2.82710005410884*rhoaxxy + 1.90405540527407*rhoaxy;

		double a0 = rho_in, ay = 0.0, ayy = 0.0, axyy = 0.0, ayyy = 0.0;
		double a[10] = {a0, ax, ay, axx, axy, ayy, axxx, axxy, axyy, ayyy};

		if(y == 5){
			printf("ax: %g axx: %g axy: %g\n", ax, axx, axy);
			printf("axxx: %g axxy: %g\n", axxx, axxy);
		}
/*
		if(y == 5){
			printf("frec\n");
			printf("f0: %g f1: %g f2: %g\n", frec[gpu_fieldn_index(x, y, 0)], frec[gpu_fieldn_index(x, y, 1)], frec[gpu_fieldn_index(x, y, 2)]);
			printf("f3: %g f4: %g f5: %g\n", frec[gpu_fieldn_index(x, y, 3)], frec[gpu_fieldn_index(x, y, 4)], frec[gpu_fieldn_index(x, y, 5)]);
			printf("f6: %g f7: %g f8: %g\n", frec[gpu_fieldn_index(x, y, 6)], frec[gpu_fieldn_index(x, y, 7)], frec[gpu_fieldn_index(x, y, 8)]);
			printf("f9: %g f10: %g f11: %g\n", frec[gpu_fieldn_index(x, y, 9)], frec[gpu_fieldn_index(x, y, 10)], frec[gpu_fieldn_index(x, y, 11)]);
			printf("f12: %g f13: %g f14: %g\n", frec[gpu_fieldn_index(x, y, 12)], frec[gpu_fieldn_index(x, y, 13)], frec[gpu_fieldn_index(x, y, 14)]);
			printf("f15: %g f16: %g\n", frec[gpu_fieldn_index(x, y, 15)], frec[gpu_fieldn_index(x, y, 16)]);
		}
*
		gpu_recursive_inlet_pressure(x, y, a, frec);
/*
		if(y == 5){
			printf("frec after\n");
			printf("f0: %g f1: %g f2: %g\n", frec[gpu_fieldn_index(x, y, 0)], frec[gpu_fieldn_index(x, y, 1)], frec[gpu_fieldn_index(x, y, 2)]);
			printf("f3: %g f4: %g f5: %g\n", frec[gpu_fieldn_index(x, y, 3)], frec[gpu_fieldn_index(x, y, 4)], frec[gpu_fieldn_index(x, y, 5)]);
			printf("f6: %g f7: %g f8: %g\n", frec[gpu_fieldn_index(x, y, 6)], frec[gpu_fieldn_index(x, y, 7)], frec[gpu_fieldn_index(x, y, 8)]);
			printf("f9: %g f10: %g f11: %g\n", frec[gpu_fieldn_index(x, y, 9)], frec[gpu_fieldn_index(x, y, 10)], frec[gpu_fieldn_index(x, y, 11)]);
			printf("f12: %g f13: %g f14: %g\n", frec[gpu_fieldn_index(x, y, 12)], frec[gpu_fieldn_index(x, y, 13)], frec[gpu_fieldn_index(x, y, 14)]);
			printf("f15: %g f16: %g\n", frec[gpu_fieldn_index(x, y, 15)], frec[gpu_fieldn_index(x, y, 16)]);
		}
*/
/*
		double rho = 0, ux_i = 0, uy_i = 0, Pxx = 0, Pxy = 0, Pyy = 0;
		for(int n = 0; n < q; ++n){
			rho += frec[gpu_fieldn_index(x, y, n)];
			ux_i += (frec[gpu_fieldn_index(x, y, n)]*ex_d[n]);
			uy_i += (frec[gpu_fieldn_index(x, y, n)]*ey_d[n]);
		}

		double ux_in = ux_i/rho;
		uy_in = uy_i/rho;

		gpu_equilibrium(x, y, rho_in, ux_in, uy_in, feq);

		double fneq;
		double tauxx = 0.0, tauxy = 0.0, tauyy = 0.0;
		for(int n = 0; n < q; ++n){
			fneq = frec[gpu_fieldn_index(x, y, n)] - feq[gpu_fieldn_index(x, y, n)];

			tauxx += (fneq)*ex_d[n]*ex_d[n];
			tauxy += (fneq)*ex_d[n]*ey_d[n];
			tauyy += (fneq)*ey_d[n]*ey_d[n];
		}
*/		
		double tauxy = axy;
		double ux_in = axxy/(2*tauxy);

		gpu_recursive(x, y, rho, ux_in, uy_in, 0.0, tauxy, 0.0, frec);

		r[gpu_scalar_index(x, y)] = rho_in;
		u[gpu_scalar_index(x, y)] = ux_in;
		v[gpu_scalar_index(x, y)] = uy_in;
		txx[gpu_scalar_index(x, y)] = tauxx;
		txy[gpu_scalar_index(x, y)] = tauxy;
		tyy[gpu_scalar_index(x, y)] = tauyy;

		if(x == 0){
			if(y == 5){
				printf("rho: %g ux: %g uy: %g\n", rho_in, ux_in, uy_in);
				printf("tauxx: %g tauxy: %g tauyy: %g\n", tauxx, tauxy, tauyy);
			}
		}

		for(int n = 0; n < q; ++n){
			f2[gpu_fieldn_index(x, y, n)] = frec[gpu_fieldn_index(x, y, n)];
		}
	}

	else if(x == 1){
		f2[gpu_fieldn_index(x, y, 9)] = f2[gpu_fieldn_index(x, y, 11)] - feq[gpu_fieldn_index(x, y, 11)] + feq[gpu_fieldn_index(x, y, 9)];
		f2[gpu_fieldn_index(x, y, 12)] = f2[gpu_fieldn_index(x, y, 10)] - feq[gpu_fieldn_index(x, y, 10)] + feq[gpu_fieldn_index(x, y, 12)];

		if(y > 0 && y < Ny_d-1){
			f2[gpu_fieldn_index(x, y, 9)] = (1.0/3.0)*f2[gpu_fieldn_index(x-1, y-1, 9)] + f2[gpu_fieldn_index(x+1, y+1, 9)] - (1.0/3.0)*f2[gpu_fieldn_index(x+2, y+2, 9)];
			f2[gpu_fieldn_index(x, y, 12)] = (1.0/3.0)*f2[gpu_fieldn_index(x-1, y+1, 12)] + f2[gpu_fieldn_index(x+1, y-1, 12)] - (1.0/3.0)*f2[gpu_fieldn_index(x+2, y-2, 12)];
		}
		f2[gpu_fieldn_index(x, y, 13)] = (1.0/3.0)*f2[gpu_fieldn_index(x-1, y, 13)] + f2[gpu_fieldn_index(x+1, y, 13)] - (1.0/3.0)*f2[gpu_fieldn_index(x+2, y, 13)];
	}

	else if(x == 2){
		f2[gpu_fieldn_index(x, y, 13)] = (1.0/6.0)*f2[gpu_fieldn_index(x-2, y, 13)] + (4.0/3.0)*f2[gpu_fieldn_index(x+1, y, 13)] - (1.0/2.0)*f2[gpu_fieldn_index(x+2, y, 13)];
	}
}

__host__ double report_convergence(unsigned int t, double *u, double *u_old, double *conv_host, double *conv_gpu, bool msg){

	double conv;
	conv = compute_convergence(u, u_old, conv_host, conv_gpu);

	if(msg){
		std::cout << std::setw(10) << t << std::setw(20) << conv << std::endl;
	}

	return conv;
}

__host__ double compute_convergence(double *u, double *u_old, double *conv_host, double *conv_gpu){

	dim3 grid(1, Ny/nThreads, 1);
	dim3 block(1, nThreads, 1);

	gpu_compute_convergence<<< grid, block, 2*block.y*sizeof(double) >>>(u, u_old, conv_gpu);
	getLastCudaError("gpu_compute_convergence kernel error");

	size_t conv_size_bytes = 2*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(conv_host, conv_gpu, conv_size_bytes, cudaMemcpyDeviceToHost));

	double convergence;
	double sumuxe2 = 0.0;
	double sumuxa2 = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		sumuxe2 += conv_host[2*i];
		sumuxa2 += conv_host[2*i+1];
	}

	convergence = sqrt(sumuxe2/sumuxa2);
	return convergence;
}

__global__ void gpu_compute_convergence(double *u, double *u_old, double *conv){

	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int x = Nx_d/2;

	extern __shared__ double data[];

	double *uxe2 = data;
	double *uxa2 = data + 1*blockDim.y;

	double ux = u[gpu_scalar_index(x, y)];
	double ux_old = u_old[gpu_scalar_index(x, y)];

	uxe2[threadIdx.y] = (ux - ux_old)*(ux - ux_old);
	uxa2[threadIdx.y] = ux_old*ux_old;

	__syncthreads();

	if(threadIdx.y == 0){

		size_t idx = 2*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 2; ++n){
			conv[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			conv[idx  ] += uxe2[i];
			conv[idx+1] += uxa2[i];
		}
	}
}

__host__ std::vector<double> report_flow_properties(unsigned int t, double conv, double *rho, double *ux, double *uy, double *prop_gpu, double *prop_host, bool msg){

	std::vector<double> prop;

	if(msg){
		prop = compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);
		std::cout << std::setw(10) << t << std::setw(13) << prop[0] << std::setw(15) << prop[1] << std::setw(20) << conv << std::endl;
	}

	return prop;
}

__host__ std::vector<double> compute_flow_properties(unsigned int t, double *r, double *u, double *v, std::vector<double> prop, double *prop_gpu, double *prop_host){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_compute_flow_properties<<< grid, block, 3*block.x*sizeof(double) >>>(t, r, u, v, prop_gpu);
	getLastCudaError("gpu_compute_flow_properties kernel error");

	size_t prop_size_bytes = 3*grid.x*grid.y*sizeof(double);
	checkCudaErrors(cudaMemcpy(prop_host, prop_gpu, prop_size_bytes, cudaMemcpyDeviceToHost));

	double E = 0.0;

	double sumuxe2 = 0.0;
	double sumuxa2 = 0.0;

	for(unsigned int i = 0; i < grid.x*grid.y; ++i){

		E += prop_host[3*i];

		sumuxe2  += prop_host[3*i+1];
		sumuxa2  += prop_host[3*i+2];
	}

	prop.push_back(E);
	prop.push_back(sqrt(sumuxe2/sumuxa2));

	return prop;
}

__global__ void gpu_compute_flow_properties(unsigned int t, double *r, double *u, double *v, double *prop_gpu){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double data[];

	double *E = data;
    double *uxe2  = data + 1*blockDim.x;
    double *uxa2  = data + 2*blockDim.x;

	double rho = r[gpu_scalar_index(x, y)];
	double ux = u[gpu_scalar_index(x, y)];
	double uy = v[gpu_scalar_index(x, y)];

	E[threadIdx.x] = rho*(ux*ux + uy*uy);

	// compute analytical results
    double uxa = poiseulle_eval(x, y, &uxa);

    // compute terms for L2 error
    uxe2[threadIdx.x]  = (ux - uxa)*(ux - uxa);
    uxa2[threadIdx.x]  = uxa*uxa;

    //printf("ux: %g uxa: %g%s\n", ux, uxa);
	//printf("uxe2: %g uxa2: %g\n", uxe2, uxa2);

	__syncthreads();

	if (threadIdx.x == 0){
		
		size_t idx = 3*(gridDim.x*blockIdx.y + blockIdx.x);

		for(int n = 0; n < 3; ++n){
			prop_gpu[idx+n] = 0.0;
		}

		for(int i = 0; i < blockDim.x; ++i){
			prop_gpu[idx  ] += E[i];
            prop_gpu[idx+1] += uxe2[i];
            prop_gpu[idx+2] += uxa2[i];
		}
	}
}

__host__ void wrapper_input(unsigned int *nx, unsigned int *ny, double *rho, double *u, double *nu, const double *tau, const double *mi_ar){
	checkCudaErrors(cudaMemcpyToSymbol(Nx_d, nx, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(Ny_d, ny, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(rho0_d, rho, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(u_max_d, u, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(nu_d, nu, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(tau_d, tau, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(mi_ar_d, mi_ar, sizeof(double)));
}

__host__ void wrapper_lattice(unsigned int *ndir, double *a, double *w_0, double *w_p, double *w_s, double *w_t, double *w_q){
	checkCudaErrors(cudaMemcpyToSymbol(q, ndir, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(as_d, a, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(w0_d, w_0, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wp_d, w_p, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(ws_d, w_s, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wt_d, w_t, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(wq_d, w_q, sizeof(double)));
}

__host__ int* generate_e(int *e, std::string mode){

	int *temp_e;

	size_t mem_e = ndir*sizeof(int);

	checkCudaErrors(cudaMalloc(&temp_e, mem_e));
	checkCudaErrors(cudaMemcpy(temp_e, e, mem_e, cudaMemcpyHostToDevice));

	if(mode == "x"){
		checkCudaErrors(cudaMemcpyToSymbol(ex_d, &temp_e, sizeof(temp_e)));
	}
	else if(mode == "y"){
		checkCudaErrors(cudaMemcpyToSymbol(ey_d, &temp_e, sizeof(temp_e)));
	}

	return temp_e;
}

__host__ bool* generate_mesh(bool *mesh, std::string mode){

	int mode_num;
	bool *temp_mesh;

	checkCudaErrors(cudaMalloc(&temp_mesh, mem_mesh));
	checkCudaErrors(cudaMemcpy(temp_mesh, mesh, mem_mesh, cudaMemcpyHostToDevice));
	

	if(mode == "walls"){
		checkCudaErrors(cudaMemcpyToSymbol(walls_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
	}

	else if(mode == "inlet"){
		checkCudaErrors(cudaMemcpyToSymbol(inlet_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 2;
	}

	else if(mode == "outlet"){
		checkCudaErrors(cudaMemcpyToSymbol(outlet_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 3;
	}

	if(meshprint){
		gpu_print_mesh<<< 1, 1 >>>(mode_num);
		printf("\n");
	}

	return temp_mesh;
}

__global__ void gpu_print_mesh(int mode){
	if(mode == 1){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", walls_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}

	else if(mode == 2){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", inlet_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}

	else if(mode == 3){
		for(int y = 0; y < Ny_d; ++y){
			for(int x = 0; x < Nx_d; ++x){
				printf("%d ", outlet_d[Nx_d*y + x]);
			}
		printf("\n");
		}
	}
}

__host__ void initialization(double *array, double value){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	gpu_initialization<<< grid, block >>>(array, value);
	getLastCudaError("gpu_print_array kernel error");
}

__global__ void gpu_initialization(double *array, double value){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	array[gpu_scalar_index(x, y)] = value;
}

__host__ bool* create_pinned_mesh(bool *array){

	bool *pinned;
	const unsigned int bytes = Nx*Ny*sizeof(bool);

	checkCudaErrors(cudaMallocHost((void**)&pinned, bytes));
	memcpy(pinned, array, bytes);
	return pinned;
}

__host__ double* create_pinned_double(){

	double *pinned;

	checkCudaErrors(cudaMallocHost((void**)&pinned, mem_size_scalar));
	return pinned;
}
