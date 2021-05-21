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

__device__ void known_moments(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *a0, double *ay, double *axx, double *axy, double *ayy, double *axxx, double *axxy, double *axyy, double *ayyy){

	double cs = 1.0/as_d;
	double cs2 = cs*cs;

	double a0_local = 0.0, ay_local = 0.0, axx_local = 0.0, axy_local = 0.0, ayy_local = 0.0, axxx_local = 0.0, axxy_local = 0.0, axyy_local = 0.0, ayyy_local = 0.0;
	for(int n = 0; n < NI; ++n){
		unsigned int ni = I[n];

		double ex2 = ex_d[ni]*ex_d[ni];
		double ey2 = ey_d[ni]*ey_d[ni];

		a0_local += f[gpu_fieldn_index(x, y, ni)];
		//ax_local += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni];
		ay_local += f[gpu_fieldn_index(x, y, ni)]*ey_d[ni];
		axx_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2);
		axy_local += f[gpu_fieldn_index(x, y, ni)]*ex_d[ni]*ey_d[ni];
		ayy_local += f[gpu_fieldn_index(x, y, ni)]*(ey2 - cs2);
		axxx_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - 3*cs2)*ex_d[ni];
		axxy_local += f[gpu_fieldn_index(x, y, ni)]*(ex2 - cs2)*ey_d[ni];
		axyy_local += f[gpu_fieldn_index(x, y, ni)]*(ey2 - cs2)*ex_d[ni];
		ayyy_local += f[gpu_fieldn_index(x, y, ni)]*(ey2 - 3*cs2)*ey_d[ni];
	}

	*a0 = a0_local;
	*ay = ay_local;
	*axx = axx_local;
	*axy = axy_local;
	*ayy = ayy_local;
	*axxx = axxx_local;
	*axxy = axxy_local;
	*axyy = axyy_local;
	*ayyy = ayyy_local;
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

__device__ void device_inlet_PP(unsigned int x, unsigned int y, double rho_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double uy_in = 0.0;
	double a0 = rho_in;

	double ax, ay, axx, axy, ayy, axxx, axxy, axyy, ayyy;

	//double rhoI = 0, rhomxx = 0, rhomxy = 0, rhomxxx = 0, rhomxxy = 0;
	//double *prhoI = &rhoI, *prhomxx = &rhomxx, *prhomxy = &rhomxy, *prhomxxx = &rhomxxx, *prhomxxy = &rhomxxy;

	double rhoI = 0, rhoay = 0, rhoaxx = 0, rhoaxy = 0, rhoayy = 0, rhoaxxx = 0, rhoaxxy = 0, rhoaxyy = 0, rhoayyy = 0;
	double *prhoI = &rhoI, *prhoay = &rhoay, *prhoaxx = &rhoaxx, *prhoaxy = &rhoaxy, *prhoayy = &rhoayy, *prhoaxxx = &rhoaxxx, *prhoaxxy = &rhoaxxy, *prhoaxyy = &rhoaxyy, *prhoayyy = &rhoayyy;

	if(x == 0){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 2, 3, 4, 6, 7, 10, 11, 14, 15, 16};

		//known_moments(x, y, NI, I, f, prhoI, prhomxx, prhomxy, prhomxxx, prhomxxy);
		known_moments(x, y, NI, I, f, prhoI, prhoay, prhoaxx, prhoaxy, prhoayy, prhoaxxx, prhoaxxy, prhoaxyy, prhoayyy);

		//rhoI = *prhoI, rhomxx = *prhomxx, rhomxy = *prhomxy, rhomxxx = *prhomxxx, rhomxxy = *prhomxxy;

		rhoI = *prhoI, rhoay = *prhoay, rhoaxx = *prhoaxx, rhoaxy = *prhoaxy, rhoayy = *prhoayy, rhoaxxx = *prhoaxxx, rhoaxxy = *prhoaxxy, rhoaxyy = *prhoaxyy, rhoayyy = *prhoayyy;
/*
		double ux = (1.02853274853573*rho_in - 1.39116351908114*rhoI - 0.148004964936173*rhomxxx - 0.94084808101913*rhomxx)/rho_in;
		double mxx = (0.740268695474152*rho_in - 0.750466883119502*rhoI + 0.384569467507977*rhomxxx + 1.12216193413231*rhomxx)/rho_in;
		double mxy = (2.57129750702885*rhomxxy + 3.73177207142366*rhomxy)/rho_in;
		double mxxx = (0.208023617035681*rho_in - 0.237543794896928*rhoI + 2.12172701123547*rhomxxx + 0.355195692599559*rhomxx)/rho_in;
		double mxxy = (2.82710005410884*rhomxxy + 1.90405540527407*rhomxy)/rho_in;
*/
		ax = 1.02853001742339*a0 - 1.3911626571536*rhoI + 0.0352938882748483*rhoay - 0.940832698105446*rhoaxx + 0.0979476254787982*rhoaxy + 0.00355923867008645*rhoayy - 0.147972332198398*rhoaxxx + 0.0916956779614975*rhoaxxy + 0.002202787016436*rhoaxyy - 0.00120944005217295*rhoayyy;
		ay = 23.5259268326181*rhoay + 62.4930443684678*rhoaxy + 58.438961358839*rhoaxxy - 0.800169075943611*rhoayyy;
		axx = 0.740263417652105*a0 - 0.750465217461107*rhoI + 0.257673935132836*rhoay + 1.12219166131709*rhoaxx + 0.715096899992883*rhoaxy + 0.00687816026470532*rhoayy + 0.384632529646006*rhoaxxx + 0.669452625650493*rhoaxxy + 0.0160821271480331*rhoaxyy - 0.00882989074832976*rhoayyy;
		axy = 23.1381663530851*rhoay + 65.1965244097511*rhoaxy + 60.0473921945547*rhoaxxy - 0.815217391304348*rhoayyy;
		ayy =  - 0.000914945321896342*a0 + 0.000288752887648945*rhoI + 8.72863616235121*rhoay + 0.0051534038878491*rhoaxx + 24.2237176905198*rhoaxy + 1.19237452524743*rhoayy + 0.010932238272953*rhoaxxx + 22.6775300118014*rhoaxxy + 0.54477778871769*rhoaxyy - 0.299110205522918*rhoayyy;
		axxx = 0.208013859147942*a0 - 0.237540715348681*rhoI + 0.158714460400339*rhoay + 0.355250653628826*rhoaxx + 0.440464490744149*rhoaxy + 0.0127166689443748*rhoayy + 2.12184360350744*rhoaxxx + 0.41234986452523*rhoaxxy + 0.00990579870282131*rhoaxyy - 0.0054387780618658*rhoayyy;
		axxy = 16.0223789279238*rhoay + 44.4653181392327*rhoaxy + 42.6271193162634*rhoaxxy - 0.549049927727359*rhoayyy;
		axyy = 16.0223789279238*rhoay + 44.4653181392327*rhoaxy + 41.6271193162634*rhoaxxy + 1.0*rhoaxyy - 0.549049927727359*rhoayyy;
		ayyy = - 0.658154004267018*rhoay - 1.81101456693753*rhoaxy- 1.64714978318208*rhoaxxy + 1.04347826086957*rhoayyy;

		//double m[10] = {1, ux, uy_in, mxx, mxy, 0, mxxx, mxxy, 0, 0};
		double a[10] = {a0, ax, ay, axx, axy, ayy, axxx, axxy, axyy, ayyy};

		//gpu_recursive_inlet_pressure(x, y, rho_in, m, frec);
		gpu_recursive_inlet_pressure(x, y, rho_in, a, frec);
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

		gpu_recursive(x, y, rho, ux_out, uy_out, tauxx, tauxy, tauyy, frec);
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

__device__ void device_outlet_PP(unsigned int x, unsigned int y, double rho_out, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy){

	double uy_out = 0.0;
	double a0 = rho_out;

	double ax, ay, axx, axy, ayy, axxx, axxy, axyy, ayyy;

	//double rhoI = 0, rhomxx = 0, rhomxy = 0, rhomxxx = 0, rhomxxy = 0;
	//double *prhoI = &rhoI, *prhomxx = &rhomxx, *prhomxy = &rhomxy, *prhomxxx = &rhomxxx, *prhomxxy = &rhomxxy;

	double rhoI = 0, rhoay = 0, rhoaxx = 0, rhoaxy = 0, rhoayy = 0, rhoaxxx = 0, rhoaxxy = 0, rhoaxyy = 0, rhoayyy = 0;
	double *prhoI = &rhoI, *prhoay = &rhoay, *prhoaxx = &rhoaxx, *prhoaxy = &rhoaxy, *prhoayy = &rhoayy, *prhoaxxx = &rhoaxxx, *prhoaxxy = &rhoaxxy, *prhoaxyy = &rhoaxyy, *prhoayyy = &rhoayyy;

	if(x == Nx_d-1){
		unsigned int NI = 11;
		unsigned int I[11] = {0, 1, 2, 4, 5, 8, 9, 12, 13, 14, 16};

		//known_moments(x, y, NI, I, f, prhoI, prhomxx, prhomxy, prhomxxx, prhomxxy);
		known_moments(x, y, NI, I, f, prhoI, prhoay, prhoaxx, prhoaxy, prhoayy, prhoaxxx, prhoaxxy, prhoaxyy, prhoayyy);

		//rhoI = *prhoI, rhomxx = *prhomxx, rhomxy = *prhomxy, rhomxxx = *prhomxxx, rhomxxy = *prhomxxy;
		rhoI = *prhoI, rhoay = *prhoay, rhoaxx = *prhoaxx, rhoaxy = *prhoaxy, rhoayy = *prhoayy, rhoaxxx = *prhoaxxx, rhoaxxy = *prhoaxxy, rhoaxyy = *prhoaxyy, rhoayyy = *prhoayyy;
/*
		double ux = (-1.02853274853573*rho_out + 1.39116351908114*rhoI - 0.148004964936173*rhomxxx + 0.94084808101913*rhomxx)/rho_out;
		double mxx = (0.740268695474152*rho_out - 0.750466883119502*rhoI - 0.384569467507977*rhomxxx + 1.12216193413231*rhomxx)/rho_out;
		double mxy = (-2.57129750702885*rhomxxy + 3.73177207142366*rhomxy)/rho_out;
		double mxxx = (-0.208023617035681*rho_out + 0.237543794896928*rhoI + 2.12172701123547*rhomxxx - 0.355195692599559*rhomxx)/rho_out;
		double mxxy = (2.82710005410884*rhomxxy - 1.90405540527407*rhomxy)/rho_out;
*/

		

		ax = 1.02853001742339*a0 - 1.3911626571536*rhoI + 0.0352938882748483*rhoay - 0.940832698105446*rhoaxx + 0.0979476254787982*rhoaxy + 0.00355923867008645*rhoayy - 0.147972332198398*rhoaxxx + 0.0916956779614975*rhoaxxy + 0.002202787016436*rhoaxyy - 0.00120944005217295*rhoayyy;
		ay = 23.5259268326181*rhoay + 62.4930443684678*rhoaxy + 58.438961358839*rhoaxxy - 0.800169075943611*rhoayyy;
		axx = 0.740263417652105*a0 - 0.750465217461107*rhoI + 0.257673935132836*rhoay + 1.12219166131709*rhoaxx + 0.715096899992883*rhoaxy + 0.00687816026470532*rhoayy + 0.384632529646006*rhoaxxx + 0.669452625650493*rhoaxxy + 0.0160821271480331*rhoaxyy - 0.00882989074832976*rhoayyy;
		axy = 23.1381663530851*rhoay + 65.1965244097511*rhoaxy + 60.0473921945547*rhoaxxy - 0.815217391304348*rhoayyy;
		ayy = - 0.000914945321896342*a0 + 0.000288752887648945*rhoI + 8.72863616235121*rhoay + 0.0051534038878491*rhoaxx + 24.2237176905198*rhoaxy + 1.19237452524743*rhoayy + 0.010932238272953*rhoaxxx + 22.6775300118014*rhoaxxy + 0.54477778871769*rhoaxyy - 0.299110205522918*rhoayyy;
		axxx = 0.208013859147942*a0 - 0.237540715348681*rhoI + 0.158714460400339*rhoay + 0.355250653628826*rhoaxx + 0.440464490744149*rhoaxy + 0.0127166689443748*rhoayy + 2.12184360350744*rhoaxxx + 0.41234986452523*rhoaxxy + 0.00990579870282131*rhoaxyy - 0.0054387780618658*rhoayyy;
		axxy = 16.0223789279238*rhoay + 44.4653181392327*rhoaxy + 42.6271193162634*rhoaxxy - 0.549049927727359*rhoayyy;
		axyy = 16.0223789279238*rhoay + 44.4653181392327*rhoaxy + 41.6271193162634*rhoaxxy + 1.0*rhoaxyy - 0.549049927727359*rhoayyy;
		ayyy = - 0.658154004267018*rhoay - 1.81101456693753*rhoaxy - 1.64714978318208*rhoaxxy + 1.04347826086957*rhoayyy;

		//double m[10] = {1, ux, uy_out, mxx, mxy, 0, mxxx, mxxy, 0, 0};
		double a[10] = {a0, ax, ay, axx, axy, ayy, axxx, axxy, axyy, ayyy};

		//gpu_recursive_inlet_pressure(x, y, rho_out, m, frec);
		gpu_recursive_inlet_pressure(x, y, rho_out, a, frec);
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

__device__ void device_corners_first(unsigned int x, unsigned int y, unsigned int NI, unsigned int *I, double *f, double *frec){

	double ux = 0.0, uy = 0.0;
	double tauxx = 0.0, tauxy = 0.0, tauyy = 0.0;

	double rhoI = 0;
	double *prhoI = &rhoI;
	known_moments(x, y, NI, I, f, prhoI);

	rhoI = *prhoI;
	double rho = 1.4971916997958*rhoI;

	gpu_recursive(x, y, rho, ux, uy, tauxx, tauxy, tauyy, frec);

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
	}
	else if(mode == "PP"){
		mode_num = 2;
	}

	gpu_inlet<<< grid, block >>>(rho_in, ux_in, f, feq, frec, r, u, v, txx, txy, tyy, mode_num);
	getLastCudaError("gpu_inlet kernel error");
}

__global__ void gpu_inlet(double rho_in, double ux_in, double *f, double *feq, double *frec, double *r, double *u, double *v, double *txx, double *txy, double *tyy, unsigned int mode_num){

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	bool node_inlet = inlet_d[gpu_scalar_index(x, y)];
	if(node_inlet){
		if(mode_num == 1){
			device_inlet_VP(x, y, ux_in, f, feq, frec, r, u, v, txx, txy, tyy);
		}
		else if(mode_num == 2){
			device_inlet_PP(x, y, rho_in, f, feq, frec, r, u, v, txx, txy, tyy);
		}
	}

	__syncthreads();
	att_moments(x, y, f, r, u, v, txx, txy, tyy);
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
			device_outlet_VP(x, y, ux_out, f, feq, frec, r, u, v, txx, txy, tyy);
		}
		else if(mode_num == 4){
			device_outlet_PP(x, y, rho_out, f, feq, frec, r, u, v, txx, txy, tyy);
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

	att_moments(x, y, f, r, u, v, txx, txy, tyy);

	// Northwest
	NI = 7;
	I[0] = 0, I[1] = 2, I[2] = 3, I[3] = 6, I[4] = 10, I[5] = 14, I[6] = 15;

	x = 0, y = Ny_d-1;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	att_moments(x, y, f, r, u, v, txx, txy, tyy);

	// Southeast
	NI = 7;
	I[0] = 0, I[1] = 1, I[2] = 4, I[3] = 8, I[4] = 12, I[5] = 13, I[6] = 16;

	x = Nx_d-1, y = 0;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	att_moments(x, y, f, r, u, v, txx, txy, tyy);

	// Northeast
	NI = 7;
	I[0] = 0, I[1] = 1, I[2] = 2, I[3] = 5, I[4] = 9, I[5] = 13, I[6] = 14;

	x = Nx_d-1, y = Ny_d-1;

	device_corners_first(x, y, NI, I, f, frec);
	__syncthreads();
	device_corners_others(x, y, f);

	att_moments(x, y, f, r, u, v, txx, txy, tyy);

}