#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuda.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include "paths.h"
#include "LBM.h"
#include "dados.h"

using namespace myGlobals;

// Input data
__constant__ unsigned int q, Nx_d, Ny_d;
__constant__ double rho0_d, u_max_d, nu_d, tau_d, mi_ar_d;

//Lattice Data
__constant__ double as_d, w0_d, wp_d, ws_d, wt_d, wq_d;
__device__ int *ex_d;
__device__ int *ey_d;

// Mesh data
__device__ bool *solid_d;

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y){
	return Nx_d*y + x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d){
	return (Nx_d*(Ny_d*(d) + y) + x);
}

__global__ void gpu_init_equilibrium(double*, double*, double*, double*);
__global__ void gpu_stream_collide_save(double*, double*, double*, double*, double*, double*, double*, double*, bool);
__global__ void gpu_compute_flow_properties(unsigned int, double*, double*, double*, double*);
__global__ void gpu_print_mesh(int);
__global__ void gpu_initialization(double*, double);

// Equilibrium
__device__ void gpu_equilibrium(unsigned int x, unsigned int y, double rho, double ux, double uy, double *feq){

	double cs = 1.0/as_d;

	double A = 1.0/(pow(cs, 2));
	double B = 1.0/(2.0*pow(cs, 4));
	double C = 1.0/(6.0*pow(cs, 6));

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	for(int n = 0; n < q; ++n){
		double order_1 = A*(ux*ex_d[n] + uy*ey_d[n]);
		double order_2 = B*(pow(ux, 2)*(pow(ex_d[n], 2) - pow(cs, 2)) + 2*ux*uy*ex_d[n]*ey_d[n] + pow(uy, 2)*(pow(ey_d[n], 2) - pow(cs, 2)));
		
		double xxx = pow(ux, 3)*(pow(ex_d[n], 3) - 3*ex_d[n]*pow(cs, 2));
		double yxx = pow(ux, 2)*uy*(pow(ex_d[n], 2)*ey_d[n] - ey_d[n]*pow(cs, 2));
		double xyy = ux*pow(uy, 2)*(ex_d[n]*pow(ey_d[n], 2) - ex_d[n]*pow(cs, 2));
		double yyy = pow(uy, 3)*(pow(ey_d[n], 3) - 3*ey_d[n]*pow(cs, 2));
		double order_3 = C*(xxx + 3*yxx + 3*xyy + yyy);

		feq[gpu_fieldn_index(x, y, n)] = W[n]*rho*(1 + order_1 + order_2 + order_3);
	}
}
// Hermites
__device__ void hermite_polynomial(int ex, int ey, double cs, double *H){

	H[0] = 1;								// 0
	H[1] = ex;								// 1 - x
	H[2] = ey;								// 1 - y
	H[3] = pow(ex, 2) - pow(cs, 2);			// 2 - xx
	H[4] = ex*ey;							// 2 - xy
	H[5] = pow(ey, 2) - pow(cs, 2);			// 2 - yy
	H[6] = pow(ex, 3) - 3*ex*pow(cs, 2);	// 3 - xxx
	H[7] = pow(ex, 2)*ey - ey*pow(cs, 2);	// 3 - yxx
	H[8] = ex*pow(ey, 2) - ex*pow(cs, 2);	// 3 - xyy
	H[9] = pow(ey, 3) - 3*ey*pow(cs, 2);	// 3 - yyy

}

__device__ void hermite_moments(double rho, double ux, double uy, double tauxx, double tauxy, double tauyy, double *a){

	a[0] = rho;											// 0
	a[1] = rho*ux;										// 1 - x
	a[2] = rho*uy;										// 1 - y
	a[3] = rho*pow(ux, 2) + tauxx;						// 2 - xx
	a[4] = rho*ux*uy + tauxy;							// 2 - xy
	a[5] = rho*pow(uy, 2) + tauyy;						// 2 - yy
	a[6] = rho*pow(ux, 3) + 3*ux*tauxx;					// 3 - xxx
	a[7] = rho*pow(ux, 2)*uy + 2*ux*tauxy + uy*tauxx;	// 3 - yxx
	a[8] = rho*ux*pow(uy, 2) + 2*uy*tauxy + ux*tauyy;	// 3 - xyy
	a[9] = rho*pow(uy, 3) + 3*uy*tauyy;					// 3 - yyy
}

// Poiseulle Flow
__device__ void poiseulle_eval(unsigned int t, unsigned int x, unsigned int y, double *u){

	double gradP = -8*u_max_d*mi_ar_d/(pow(Ny_d, 2) - 2*Ny_d);

	double ux = (-1/(2*mi_ar_d))*(gradP)*((Ny_d - 1)*y - pow(y, 2));

	*u = ux;
}

// Boundary Conditions
__device__ void gpu_bounce_back(unsigned int x, unsigned int y, double *f){
	
	if(y == 0){
		f[gpu_fieldn_index(x, y, 2)] = f[gpu_fieldn_index(x, y, 4)];
		f[gpu_fieldn_index(x, y, 5)] = f[gpu_fieldn_index(x, y, 7)];
		f[gpu_fieldn_index(x, y, 6)] = f[gpu_fieldn_index(x, y, 8)];

		//f[gpu_fieldn_index(x, y, 9)] = f[gpu_fieldn_index(x+1, y+1, 11)];
		//f[gpu_fieldn_index(x, y, 10)] = f[gpu_fieldn_index(x-1, y+1, 12)];

		//f[gpu_fieldn_index(x, y, 14)] = f[gpu_fieldn_index(x, y+2, 16)];

		f[gpu_fieldn_index(x+1, y+1, 9)] = f[gpu_fieldn_index(x, y, 11)];
		f[gpu_fieldn_index(x-1, y+1, 10)] = f[gpu_fieldn_index(x, y, 12)];

		f[gpu_fieldn_index(x, y+2, 14)] = f[gpu_fieldn_index(x, y, 16)];
	}

	if(y == Ny_d-1){
		f[gpu_fieldn_index(x, y, 4)] = f[gpu_fieldn_index(x, y, 2)];
		f[gpu_fieldn_index(x, y, 7)] = f[gpu_fieldn_index(x, y, 5)];
		f[gpu_fieldn_index(x, y, 8)] = f[gpu_fieldn_index(x, y, 6)];

		//f[gpu_fieldn_index(x, y, 11)] = f[gpu_fieldn_index(x-1, y-1, 9)];
		//f[gpu_fieldn_index(x, y, 12)] = f[gpu_fieldn_index(x+1, y-1, 10)];

		//f[gpu_fieldn_index(x, y, 16)] = f[gpu_fieldn_index(x, y-2, 14)];

		f[gpu_fieldn_index(x-1, y-1, 11)] = f[gpu_fieldn_index(x, y, 9)];
		f[gpu_fieldn_index(x+1, y-1, 12)] = f[gpu_fieldn_index(x, y, 10)];

		f[gpu_fieldn_index(x, y-2, 16)] = f[gpu_fieldn_index(x, y, 14)];
	}
}

__device__ void gpu_PPBC_inlet(unsigned int x, unsigned int y, double *u, double *v, double *f, double *feq, double *feq_aux){

	double cs = 1.0/as_d;

	// Variables to periodic condition with pressure variation
	double gradP = -8*u_max_d*mi_ar_d/(pow(Ny_d, 2) - 2*Ny_d);
	double gradRho = (Nx_d/(pow(cs, 2)))*gradP;

	double rho_in = rho0_d;
	double rho_out = rho_in + gradRho;

	double ux = u[gpu_scalar_index(Nx_d-1 - x, y)];
	double uy = v[gpu_scalar_index(Nx_d-1 - x, y)];

	for(int n = 0; n < q; ++n){
		gpu_equilibrium(x, y, rho_in, ux, uy, feq_aux);
		f[gpu_fieldn_index(x, y, n)] = feq_aux[n] + (f[gpu_fieldn_index(Nx_d-1 - x, y, n)] - feq[gpu_fieldn_index(Nx_d-1 - x, y, n)]);
	}
}

__device__ void gpu_PPBC_outlet(unsigned int x, unsigned int y, double *u, double *v, double *f, double *feq, double *feq_aux){

	double cs = 1.0/as_d;

	// Variables to periodic condition with pressure variation
	double gradP = -8*u_max_d*mi_ar_d/(pow(Ny_d, 2) - 2*Ny_d);
	double gradRho = (Nx_d/(pow(cs, 2)))*gradP;

	double rho_in = rho0_d;
	double rho_out = rho_in + gradRho;

	double ux = u[gpu_scalar_index(Nx_d-1 - x, y)];
	double uy = v[gpu_scalar_index(Nx_d-1 - x, y)];

	for(int n = 0; n < q; ++n){
			gpu_equilibrium(x, y, rho_out, ux, uy, feq_aux);
			f[gpu_fieldn_index(x, y, n)] = feq_aux[n] + (f[gpu_fieldn_index(Nx_d-1 - x, y, n)] - feq[gpu_fieldn_index(Nx_d-1 - x, y, n)]);	// Periodic with pressure Outlet
		}
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

__host__ void stream_collide_save(double *f1, double *f2, double *f1rec, double *feq, double *feq_aux, double *r, double *u, double *v, bool save){

	dim3 grid(Nx/nThreads, Ny, 1);
	dim3 block(nThreads, 1, 1);

	//dim3 grid(1,1,1);
	//dim3 block(1,1,1);

	gpu_stream_collide_save<<< grid, block >>>(f1, f2, f1rec, feq, feq_aux, r, u, v, save);
	getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_stream_collide_save(double *f1, double *f2, double *f1rec, double *feq, double *feq_aux, double *r, double *u, double *v, bool save){

	const double omega = 1.0/tau_d;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int xf, yf, xb, yb;

	// Streaming Step
	// 1 - 8 directions
	xf = (x + 1)%Nx_d;		// Forward
	yf = (y + 1)%Ny_d;		// Forward
	xb = (Nx_d + x - 1)%Nx_d;	// Backward
	yb = (Ny_d + y - 1)%Ny_d; // Backward

	double ft0 = f1[gpu_fieldn_index(x, y, 0)];
	double ft1 = f1[gpu_fieldn_index(xb, y, 1)];
	double ft2 = f1[gpu_fieldn_index(x, yb, 2)];
	double ft3 = f1[gpu_fieldn_index(xf, y, 3)];
	double ft4 = f1[gpu_fieldn_index(x, yf, 4)];
	double ft5 = f1[gpu_fieldn_index(xb, yb, 5)];
	double ft6 = f1[gpu_fieldn_index(xf, yb, 6)];
	double ft7 = f1[gpu_fieldn_index(xf, yf, 7)];
	double ft8 = f1[gpu_fieldn_index(xb, yf, 8)];

	// 9 - 12 directions
	xf = (x + 2)%Nx_d;		// Forward
	yf = (y + 2)%Ny_d;		// Forward
	xb = (Nx_d + x - 2)%Nx_d;	// Backward
	yb = (Ny_d + y - 2)%Ny_d; // Backward

	double ft9 = f1[gpu_fieldn_index(xb, yb, 9)];
	double ft10 = f1[gpu_fieldn_index(xf, yb, 10)];
	double ft11 = f1[gpu_fieldn_index(xf, yf, 11)];
	double ft12 = f1[gpu_fieldn_index(xb, yf, 12)];

	// 13 - 16 directions
	xf = (x + 3)%Nx_d;		// Forward
	yf = (y + 3)%Ny_d;		// Forward
	xb = (Nx_d + x - 3)%Nx_d;	// Backward
	yb = (Ny_d + y - 3)%Ny_d; // Backward

	double ft13 = f1[gpu_fieldn_index(xb, y, 13)];
	double ft14 = f1[gpu_fieldn_index(x, yb, 14)];
	double ft15 = f1[gpu_fieldn_index(xf, y, 15)];
	double ft16 = f1[gpu_fieldn_index(x, yf, 16)];

	double f[] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13, ft14, ft15, ft16};
	double rho = 0, ux_i = 0, uy_i = 0, tau_xx = 0, tau_xy = 0, tau_yy = 0;

	for(int n = 0; n < q; ++n){
		rho += f[n];
		ux_i += f[n]*ex_d[n];
		uy_i += f[n]*ey_d[n];
		tau_xx += f[n]*ex_d[n]*ex_d[n];
		tau_xy += f[n]*ex_d[n]*ey_d[n];
		tau_yy += f[n]*ey_d[n]*ey_d[n];
	}

	double ux = ux_i/rho;
	double uy = uy_i/rho;

	if(save){
		r[gpu_scalar_index(x, y)] = rho;
		u[gpu_scalar_index(x, y)] = ux;
		v[gpu_scalar_index(x, y)] = uy;
	}

	double cs = 1.0/as_d;

	double A = 1.0/(pow(cs, 2));
	double B = 1.0/(2.0*pow(cs, 4));
	double C = 1.0/(6.0*pow(cs, 6));

	double W[] = {w0_d, wp_d, wp_d, wp_d, wp_d, ws_d, ws_d, ws_d, ws_d, wt_d, wt_d, wt_d, wt_d, wq_d, wq_d, wq_d, wq_d};

	// Calculating the regularized recursive distribution
	double a[10], H[10];
	for(int n = 0; n < q; ++n){
		hermite_polynomial(ex_d[n], ey_d[n], cs, H);
		hermite_moments(rho, ux, uy, tau_xx, tau_xy, tau_yy, a);

		//					f 			  = W *  (   0      + A*(    x     +     y)     + B*(    xx    +     xy/yx   +    yy)     + C*(   xxx    +    yxx    +    xyy    +    yyy))
		f1rec[gpu_fieldn_index(x, y, n)] = W[n]*(a[0]*H[0] + A*(a[1]*H[1] + a[2]*H[2]) + B*(a[3]*H[3] + 2*a[4]*H[4] + a[5]*H[5]) + C*(a[6]*H[6] + 3*a[7]*H[7] + 3*a[8]*H[8] + a[9]*H[9]));
	}

	// Collision Step
	for(int n = 0; n < q; ++n){
		gpu_equilibrium(x, y, rho, ux, uy, feq);
		f2[gpu_fieldn_index(x, y, n)] = omega*feq[n] + (1 - omega)*f1rec[gpu_fieldn_index(x, y, n)];
	}
/*
	if(x == 0){
		gpu_PPBC_inlet(x, y, u, v, f2, feq, feq_aux);
	}

	if(x == 1){
		gpu_PPBC_inlet(x, y, u, v, f2, feq, feq_aux);
	}

	if(x == 2){
		gpu_PPBC_inlet(x, y, u, v, f2, feq, feq_aux);
	}

	if(x == Nx_d-1){
		gpu_PPBC_outlet(x, y, u, v, f2, feq, feq_aux);
	}

	if(x == Nx_d-2){
		gpu_PPBC_outlet(x, y, u, v, f2, feq, feq_aux);
	}

	if(x == Nx_d-3){
		gpu_PPBC_outlet(x, y, u, v, f2, feq, feq_aux);
	}
*/
	bool node_solid = solid_d[gpu_scalar_index(x, y)];

	// Applying Boundary Conditions
	if(node_solid){
		gpu_bounce_back(x, y, f2);
	}

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
    double uxa;
    poiseulle_eval(t, x, y, &uxa);

     // compute terms for L2 error
    uxe2[threadIdx.x]  = (ux - uxa)*(ux - uxa);
    uxa2[threadIdx.x]  = uxa*uxa;

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

__host__ std::vector<double> report_flow_properties(unsigned int t, double *rho, double *ux, double *uy,
									 double *prop_gpu, double *prop_host, bool msg, bool computeFlowProperties){

	std::vector<double> prop;
	prop = compute_flow_properties(t, rho, ux, uy, prop, prop_gpu, prop_host);

	if(msg){
		if(computeFlowProperties){
			printf("%u, %g, %g\n", t, prop[0], prop[1]);
		}

		if(!quiet){
			printf("Completed timestep %d\n", t);
		}
	}
	
	return prop;
}

__host__ void save_scalar(const std::string name, double *scalar_gpu, double *scalar_host, unsigned int n){

	std::ostringstream path, filename;

	std::string ext = ".dat";

	int ndigits = floor(log10((double)NSTEPS) + 1.0);

	const char* path_results_c = strdup(folder.c_str());

	DIR *dir_results = opendir(path_results_c);
	if(ENOENT == errno){
		mkdir(path_results_c, ACCESSPERMS);
	}

	closedir(dir_results);

	path << folder << name << "/";
	const char* path_c = strdup(path.str().c_str());

	DIR *dir = opendir(path_c);
	if(ENOENT == errno){
		mkdir(path_c, ACCESSPERMS);
	}

	closedir(dir);

	filename << path.str() << name << std::setfill('0') << std::setw(ndigits) << n << ext;
	const char* filename_c = strdup(filename.str().c_str());

	checkCudaErrors(cudaMemcpy(scalar_host, scalar_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	FILE* fout = fopen(filename_c, "wb+");

	fwrite(scalar_host, 1, mem_size_scalar, fout);

	if(ferror(fout)){
		fprintf(stderr, "Error saving to %s\n", filename_c);
		perror("");
	}
	else{
		if(!quiet){
			printf("Saved to %s\n", filename_c);
		}
	}
	fclose(fout);
}

void wrapper_input(unsigned int *nx, unsigned int *ny, double *rho, double *u, double *nu, const double *tau, const double *mi_ar){
	checkCudaErrors(cudaMemcpyToSymbol(Nx_d, nx, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(Ny_d, ny, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(rho0_d, rho, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(u_max_d, u, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(nu_d, nu, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(tau_d, tau, sizeof(double)));
	checkCudaErrors(cudaMemcpyToSymbol(mi_ar_d, mi_ar, sizeof(double)));
}

void wrapper_lattice(unsigned int *ndir, double *a, double *w_0, double *w_p, double *w_s, double *w_t, double *w_q){
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
	

	if(mode == "solid"){
		checkCudaErrors(cudaMemcpyToSymbol(solid_d, &temp_mesh, sizeof(temp_mesh)));
		mode_num = 1;
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
				printf("%d ", solid_d[Nx_d*y + x]);
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
